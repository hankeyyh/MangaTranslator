"""
MangaTranslator jobs + SSE 压测脚本。

目标接口：
  POST {url}/v1/jobs
  GET  {url}/v1/jobs/{job_id}/events

鉴权：Authorization: Bearer $MT_API_KEY

SSE 事件：
  queued           入队 → 记 queue_seen（不算错误），记下排队起点
  started          Worker 接手 → 与 queued 配对算 queue wait
  image_progress   单图阶段（一期只有 stage=start）
  image_completed  单图成功 → 与 start 配对算 image latency
  image_failed     单图失败 → 单独计数，请求可能仍继续
  job_completed    整批完成 → stream_ok
  job_failed       整批失败 → stream_error
  heartbeat        忽略
  断连且未见终态   → stream_error
  超过 --timeout   → 取消 reader，记 timeout
  HTTP 非 2xx / 无 body → http_fail
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import time
import uuid
from pathlib import Path

import httpx

BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
DEFAULT_IMAGES = BENCH_DIR / "fixtures"
DEFAULT_CONFIG = BENCH_DIR / "configs/no_save.json"
# 压测禁止落最终图：worker 见 output.type=none 时不写盘、不上传、不 commit Volume
BENCH_OUTPUT = {"type": "none"}
LOCAL_URL = "http://127.0.0.1:8000"
REMOTE_URL = "https://hankeyyh--manga-translator-mt-web.modal.run"
PROD_ENV_FILE = PROJECT_ROOT / ".env.prod"

# 与 deploy/modal_config.MAX_IMAGES_PER_JOB 对齐
MAX_IMAGES_PER_JOB = 20
IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}
CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".avif": "image/avif",
}


def existing_dir(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {value}")
    return path


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {value}")
    return path


def positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"value must be positive int: {value}")
    return n


def _read_dotenv_key(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("'").strip('"')
    return ""


def resolve_api_key(explicit: str) -> str:
    key = (explicit or "").strip() or os.environ.get("MT_API_KEY", "").strip()
    if not key:
        key = _read_dotenv_key(PROD_ENV_FILE, "MT_API_KEY")
    if not key:
        raise SystemExit(
            "MT_API_KEY is required（--api-key / 环境变量 / .env.prod）"
        )
    return key


def validate_bench_config(config: object, config_path: Path) -> dict:
    if not isinstance(config, dict):
        raise SystemExit(f"{config_path}: config 必须是 JSON object")

    # 允许直接放 JobConfigIn，自动包一层 output.type=none
    if "config" not in config and (
        "provider" in config or "input_language" in config
    ):
        config = {"output": dict(BENCH_OUTPUT), "config": config}

    output = config.get("output")
    if not isinstance(output, dict):
        raise SystemExit(
            f"{config_path}: 压测必须设置 output.type=none"
            f"（worker 据此跳过最终图写盘 / 上传）"
        )
    output_type = output.get("type")
    if output_type != "none":
        raise SystemExit(
            f"{config_path}: 压测禁止落盘或对象存储，output.type 必须是 none，"
            f"当前={output_type!r}"
        )
    config["output"] = dict(BENCH_OUTPUT)
    inner = config.get("config")
    if inner is not None and not isinstance(inner, dict):
        raise SystemExit(f"{config_path}: config 必须是 JSON object")
    return config


def compute_time_percent(latencies: list[float], percent: float) -> float | None:
    if not latencies:
        return None
    ordered = sorted(latencies)
    n = len(ordered)
    rank = min(n, max(1, (n * percent).__ceil__()))
    return ordered[rank - 1]


def print_time_percent(name: str, samples: list[float]) -> None:
    print(
        "{} p50={:.3f}s p90={:.3f}s p95={:.3f}s (n={})".format(
            name,
            compute_time_percent(samples, 0.5) or 0,
            compute_time_percent(samples, 0.9) or 0,
            compute_time_percent(samples, 0.95) or 0,
            len(samples),
        )
    )


def log_req(worker: int, query: int, msg: str) -> None:
    print(f"[w{worker}#{query}] {msg}", flush=True)


def _since(t0: float, now: float | None = None) -> str:
    return f"+{(now if now is not None else time.monotonic()) - t0:.1f}s"


def _content_type(name: str) -> str:
    return CONTENT_TYPES.get(Path(name).suffix.lower(), "application/octet-stream")


def parse_sse_block(block: str) -> tuple[str | None, dict]:
    event_name = None
    data_raw = None
    for line in block.splitlines():
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_raw = line[len("data:") :].strip()
    if not event_name:
        return None, {}
    if event_name == "heartbeat":
        return event_name, {}
    if not data_raw:
        return event_name, {}
    try:
        payload = json.loads(data_raw)
    except json.JSONDecodeError:
        return event_name, {"raw": data_raw}
    if isinstance(payload, dict):
        return event_name, payload
    return event_name, {"data": payload}


def build_payload(
    template: dict,
    job_id: str,
    batch_len: int,
    worker_idx: int,
    query_i: int,
) -> dict:
    payload = copy.deepcopy(template)
    payload["job_id"] = job_id
    payload["images"] = [
        {"image_id": f"{worker_idx}-{query_i}-{image_idx}", "index": image_idx}
        for image_idx in range(batch_len)
    ]
    payload["output"] = dict(BENCH_OUTPUT)
    return payload


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MangaTranslator jobs + SSE 压测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--local",
        action="store_true",
        help=f"打本地 {LOCAL_URL}",
    )
    target.add_argument(
        "--remote",
        action="store_true",
        help=f"打 Modal {REMOTE_URL}",
    )
    target.add_argument(
        "--url",
        help="自定义 gateway base URL",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Bearer token；默认读 MT_API_KEY，再回退 .env.prod",
    )
    parser.add_argument(
        "--images",
        default=str(DEFAULT_IMAGES),
        type=existing_dir,
        help="图片目录；按 --batch-size 切片，不足则循环复用",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        type=existing_file,
        help="压测 JSON 路径；必须 output.type=none",
    )
    parser.add_argument(
        "--concurrency",
        default=1,
        type=positive_int,
        help="同时 in-flight 的 job 数（压测旋钮）",
    )
    parser.add_argument(
        "--batch-size",
        default=2,
        type=positive_int,
        help="每发几张图",
    )
    parser.add_argument(
        "--duration", default=300, type=positive_int, help="墙钟时长，单位秒"
    )
    parser.add_argument(
        "--timeout",
        default=180,
        type=positive_int,
        help="单次 submit+SSE 上限秒数；超时记 timeout 并取消 reader",
    )
    args = parser.parse_args()
    if args.batch_size > MAX_IMAGES_PER_JOB:
        raise SystemExit(
            f"--batch-size {args.batch_size} 超过 MAX_IMAGES_PER_JOB={MAX_IMAGES_PER_JOB}"
        )
    if args.local:
        args.url = LOCAL_URL
    elif args.remote:
        args.url = REMOTE_URL
    args.api_key = resolve_api_key(args.api_key)
    return args


async def worker(
    client: httpx.AsyncClient,
    idx: int,
    args: argparse.Namespace,
    image_batchs: list[list[tuple[str, bytes]]],
    config: dict,
    stats: dict,
) -> None:
    stop_at = time.monotonic() + args.duration
    i = idx
    while time.monotonic() < stop_at:
        batch = image_batchs[i % len(image_batchs)]
        names = ",".join(name for name, _ in batch)
        remain = max(0.0, stop_at - time.monotonic())
        log_req(
            idx,
            i,
            f"start n={len(batch)} images={names} remain={remain:.0f}s",
        )
        t0 = time.monotonic()
        kind = await run_once(client, args, batch, config, stats, idx, i)
        log_req(idx, i, f"finish {kind} wall={time.monotonic() - t0:.1f}s")
        i += 1


async def run_once(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    images: list[tuple[str, bytes]],
    config: dict,
    stats: dict,
    worker_idx: int,
    query_i: int,
) -> str:
    base = args.url.rstrip("/")
    job_id = f"bench-{worker_idx}-{query_i}-{uuid.uuid4().hex[:8]}"
    payload = build_payload(config, job_id, len(images), worker_idx, query_i)
    image_ids = ",".join(item["image_id"] for item in payload["images"])
    got_sse = False
    try:
        async with asyncio.timeout(args.timeout):
            submit_at = time.monotonic()
            queue_start = 0.0
            image_starts: dict[str, float] = {}
            stats["counters"]["submitted"] += 1
            response = await client.post(
                f"{base}/v1/jobs",
                headers={"Authorization": f"Bearer {args.api_key}"},
                data={"payload": json.dumps(payload, ensure_ascii=False)},
                files=[
                    ("images", (name, data, _content_type(name)))
                    for name, data in images
                ],
            )
            header_at = time.monotonic()
            header_latency = header_at - submit_at
            stats["header_latencies"].append(header_latency)
            log_req(
                worker_idx,
                query_i,
                f"{_since(submit_at, header_at)} header status={response.status_code} "
                f"job={job_id} ids={image_ids}",
            )
            if response.status_code not in (200, 202):
                stats["counters"]["http_fail"] += 1
                log_req(
                    worker_idx,
                    query_i,
                    f"{_since(submit_at)} http_fail status={response.status_code} "
                    f"body={response.text[:200]}",
                )
                return "http_fail"
            try:
                created = response.json()
                job_id = str(created.get("job_id") or job_id)
            except json.JSONDecodeError:
                pass

            async with client.stream(
                "GET",
                f"{base}/v1/jobs/{job_id}/events",
                headers={
                    "Authorization": f"Bearer {args.api_key}",
                    "Accept": "text/event-stream",
                },
            ) as sse:
                if sse.status_code != 200:
                    stats["counters"]["http_fail"] += 1
                    log_req(
                        worker_idx,
                        query_i,
                        f"{_since(submit_at)} http_fail sse status={sse.status_code}",
                    )
                    return "http_fail"

                buf = ""
                async for chunk in sse.aiter_text():
                    if chunk:
                        got_sse = True
                    buf += chunk
                    while "\n\n" in buf:
                        block, buf = buf.split("\n\n", 1)
                        name, data = parse_sse_block(block)
                        if not name or name == "heartbeat":
                            continue
                        now = time.monotonic()
                        kind = _handle_event(
                            name,
                            data,
                            now,
                            submit_at,
                            header_latency,
                            queue_start,
                            image_starts,
                            stats,
                            worker_idx,
                            query_i,
                        )
                        if name == "queued" and queue_start == 0.0:
                            queue_start = now
                        if kind is not None:
                            return kind
                if got_sse:
                    stats["counters"]["stream_error"] += 1
                    log_req(
                        worker_idx,
                        query_i,
                        f"{_since(submit_at)} stream_error closed before terminal",
                    )
                    return "stream_error"
                stats["counters"]["http_fail"] += 1
                log_req(worker_idx, query_i, f"{_since(submit_at)} http_fail no body")
                return "http_fail"
    except httpx.HTTPError as e:
        if got_sse:
            stats["counters"]["stream_error"] += 1
            log_req(worker_idx, query_i, f"stream_error {e}")
            return "stream_error"
        stats["counters"]["http_fail"] += 1
        log_req(worker_idx, query_i, f"http_fail {e}")
        return "http_fail"
    except TimeoutError:
        stats["counters"]["timeout"] += 1
        log_req(worker_idx, query_i, f"timeout after {args.timeout}s")
        return "timeout"


def _handle_event(
    name: str,
    data: dict,
    now: float,
    submit_at: float,
    header_latency: float,
    queue_start: float,
    image_starts: dict[str, float],
    stats: dict,
    worker_idx: int,
    query_i: int,
) -> str | None:
    if name == "queued":
        pos = data.get("position", "")
        if queue_start == 0.0:
            stats["counters"]["queue_seen"] += 1
            log_req(
                worker_idx,
                query_i,
                f"{_since(submit_at, now)} queue pos={pos}",
            )
        else:
            log_req(
                worker_idx,
                query_i,
                f"{_since(submit_at, now)} queue pos={pos}",
            )
        return None

    if name == "started":
        if queue_start != 0.0:
            queue_wait = now - queue_start
            stats["queue_waits"].append(queue_wait)
            log_req(
                worker_idx,
                query_i,
                f"{_since(submit_at, now)} started queue_wait={queue_wait:.2f}s "
                f"header={header_latency:.2f}s worker={data.get('worker_id', '')}",
            )
        else:
            stats["queue_waits"].append(now - submit_at)
            log_req(
                worker_idx,
                query_i,
                f"{_since(submit_at, now)} started worker={data.get('worker_id', '')}",
            )
        return None

    if name == "image_progress":
        image_id = str(data.get("image_id") or "")
        stage = str(data.get("stage") or "")
        if image_id and image_id not in image_starts:
            image_starts[image_id] = now
        log_req(
            worker_idx,
            query_i,
            f"{_since(submit_at, now)} image_progress {image_id} stage={stage}",
        )
        return None

    if name == "image_completed":
        image_id = str(data.get("image_id") or "")
        t0 = image_starts.pop(image_id, None)
        if t0 is not None:
            elapsed = now - t0
            stats["image_latencies"].append(elapsed)
            log_req(
                worker_idx,
                query_i,
                f"{_since(submit_at, now)} image_completed {image_id} "
                f"image={elapsed:.2f}s",
            )
        else:
            log_req(
                worker_idx,
                query_i,
                f"{_since(submit_at, now)} image_completed {image_id}",
            )
        return None

    if name == "image_failed":
        stats["counters"]["image_failed"] += 1
        image_id = str(data.get("image_id") or "")
        image_starts.pop(image_id, None)
        log_req(
            worker_idx,
            query_i,
            f"{_since(submit_at, now)} image_failed {image_id} "
            f"error={data.get('error', '')}",
        )
        return None

    if name == "job_completed":
        stats["counters"]["stream_ok"] += 1
        stats["latencies"].append(now - submit_at)
        log_req(
            worker_idx,
            query_i,
            f"{_since(submit_at, now)} job_completed "
            f"ok={data.get('success_count')} fail={data.get('error_count')} "
            f"latency={now - submit_at:.2f}s",
        )
        return "ok"

    if name == "job_failed":
        stats["counters"]["stream_error"] += 1
        log_req(
            worker_idx,
            query_i,
            f"{_since(submit_at, now)} stream_error job_failed "
            f"error={data.get('error', '')}",
        )
        return "stream_error"

    log_req(worker_idx, query_i, f"{_since(submit_at, now)} {name} {data}")
    return None


async def benchpress(
    args: argparse.Namespace,
    image_batchs: list[list[tuple[str, bytes]]],
    config: dict,
) -> None:
    concurrency: int = args.concurrency
    limits = httpx.Limits(
        max_connections=concurrency * 2,
        max_keepalive_connections=concurrency * 2,
    )

    stats = {
        "counters": {
            "submitted": 0,
            "http_fail": 0,
            "stream_ok": 0,
            "stream_error": 0,
            "image_failed": 0,
            "timeout": 0,
            "queue_seen": 0,
        },
        "latencies": [],
        "header_latencies": [],
        "queue_waits": [],
        "image_latencies": [],
    }

    try:
        async with httpx.AsyncClient(limits=limits, timeout=None) as client:
            workers = [
                worker(client, i, args, image_batchs, copy.deepcopy(config), stats)
                for i in range(concurrency)
            ]
            await asyncio.gather(*workers)
    except asyncio.CancelledError:
        print("Shutdown by Ctrl+C")
    finally:
        counters = stats["counters"]
        print("[counters]", counters)
        print("[concurrency]", concurrency)
        submitted = counters["submitted"]
        n_err = counters["http_fail"] + counters["stream_error"] + counters["timeout"]
        error_rate = (n_err / submitted * 100) if submitted else 0
        print(
            "[error_rate]={:.2f}% (http_fail+stream_error+timeout={}, submitted={})".format(
                error_rate, n_err, submitted
            )
        )
        print_time_percent("[whole_latency]", stats["latencies"])
        print_time_percent("[header]", stats["header_latencies"])
        print_time_percent("[queue_wait]", stats["queue_waits"])
        print_time_percent("[image]", stats["image_latencies"])


if __name__ == "__main__":
    args = parse_arguments()
    image_dir: Path = args.images
    batch_size: int = args.batch_size

    image_paths: list[Path] = []
    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIX:
            image_paths.append(p)

    if len(image_paths) == 0:
        raise SystemExit(f"no images in {image_dir}")

    image_paths = sorted(image_paths)
    images = [(p.name, p.read_bytes()) for p in image_paths]

    image_batchs: list[list[tuple[str, bytes]]] = []
    image_len = len(images)
    for i in range(0, image_len, batch_size):
        image_batchs.append([images[(i + j) % image_len] for j in range(batch_size)])

    with args.config.open(encoding="utf-8") as f:
        config = json.load(f)

    config = validate_bench_config(config, args.config)

    asyncio.run(benchpress(args, image_batchs, config))
