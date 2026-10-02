# bench — MangaTranslator jobs + SSE 压测

直打已部署（或本地）的 `POST /v1/jobs` + `GET /v1/jobs/{id}/events`，观察 **容量**：容器是否扩容、排队到 `started` 要多久、错误如何分类。

禁止把本目录挂进 pytest 或 `deploy.sh test`。一次压测会占 GPU、按墙钟计费。

## 打哪里

```
POST {BASE_URL}/v1/jobs
Authorization: Bearer $MT_API_KEY
Content-Type: multipart/form-data
  payload: JSON 字符串（CreateJobRequest，images 只带 image_id/index）
  images: 若干文件（与 payload.images 等长）

GET {BASE_URL}/v1/jobs/{job_id}/events
Authorization: Bearer $MT_API_KEY
Accept: text/event-stream
```

生产 Modal URL 形如 `https://<user>--manga-translator-mt-web.modal.run`。本地可对 `http://127.0.0.1:8000`，但本地测不到 Modal 扩容。

响应是 SSE，与 `deploy/smoke_test.py` 相同：

```
id: <seq>
event: <name>
data: <json>

```

| event | 含义 | 压测怎么记 |
|--------|------|------------|
| `queued` | 入队（创建时就会写） | `queue_seen`；记下排队起点。**不是错误** |
| `started` | Worker 开始处理 | 与 `queued` 配对，算 queue wait |
| `image_progress` | 单图阶段（一期只有 `stage=start`） | 记下该图开始时刻 |
| `image_completed` | 单图成功 | 与 `start` 配对，算 image latency |
| `image_failed` | 单图失败 | `image_failed` 单独计数；请求可能仍继续 |
| `job_completed` | 整批结束 | `stream_ok`，结束本请求 |
| `job_failed` | 整批失败 | `stream_error`，结束本请求 |
| `heartbeat` | 保活 | 忽略 |
| 连接断开且未见终态 | SSE 提前结束 | `stream_error` |
| 超过 `--timeout` | 取消 reader | `timeout` |
| HTTP 非 2xx / 无 body | 提交或拉流失败 | `http_fail` |

终态：`job_completed` / `job_failed`。

## 发压模型

固定并发闭环，不是按 QPS 发射：

1. 启动 `concurrency` 个 worker。
2. 每个 worker：`POST /v1/jobs` → 堵住读 SSE 直到终态/超时 → **立刻**再提交下一批。
3. 直到 `duration` 到时；不再发新请求，收尾 in-flight。

任意时刻 in-flight ≈ `concurrency`。并发单位是 **job 请求**，不是图片张数。

对照当前 Modal 配置（`deploy/modal_config.py`）：

- GPU Worker：A10G / 4 CPU / 16GB，`max_inputs=2`（一容器最多同时 2 个 job）
- `min_containers = 0`：空闲会缩到 0，压测开头会有冷启动
- `MAX_IMAGES_PER_JOB = 20`：`--batch-size` 不得超过 20
- Worker timeout = 900s；脚本 `--timeout` 必须小于它，否则 hang 时 Grafana 会看起来空闲

建议阶梯（每次只改 concurrency，图片和 config 固定）：

```
phase 0: concurrency=1, duration≥2min   基线延迟、确认协议
phase 1: concurrency=2                  看是否复用同一容器（max_inputs=2）
phase 2: concurrency=4 / 8              看扩容、排队、错误率
```

脚本只输出计数和延迟分位。容器数、GPU、OOM 看 Modal Dashboard / Grafana，按时间轴对齐。

## 目录约定

```
bench/
  README.md
  loadtest_jobs.py        # 发压脚本
  configs/                # 压测专用 payload，不要复用 supabase/volume 保存配置
  fixtures/               # 固定图片集；也可 --images 指向仓库里已有漫画页
```

`config` 必须满足服务端校验：

- `len(images) == len(payload.images)`
- **必须** `output.type = none`。脚本会强制写入该字段。Worker 见到 `none` 时：不把成图交给 `translate_and_render` 落盘、不拷到 Volume、不上传 Supabase、不 `commit` scratch。`volume` / `supabase` 才会保存最终图。
- provider / detector / ocr 选型当作实验维度，一次压测内固定，否则 Grafana 说不清是负载还是模型变了。

每发请求应改写 `job_id` 和 `image_id`（带 worker/轮次），避免多 worker 撞同一标识。

## 怎么跑

在 `MangaTranslator` 仓库根目录（需要 `httpx`）：

```bash
export MT_API_KEY=...   # 也可 --api-key，或从 .env.prod 自动读

python bench/loadtest_jobs.py \
  --remote \
  --images bench/fixtures \
  --config bench/configs/no_save.json \
  --concurrency 2 \
  --batch-size 2 \
  --duration 60 \
  --timeout 180
```

本地则把 `--remote` 换成 `--local`（`http://127.0.0.1:8000`）。`--local` / `--remote` / `--url` 必须三选一。

| 参数 | 含义 |
|------|------|
| `--local` | 打本地 `http://127.0.0.1:8000` |
| `--remote` | 打 Modal `https://hankeyyh--manga-translator-mt-web.modal.run` |
| `--url` | 自定义 gateway base URL |
| `--api-key` | Bearer token；默认 `MT_API_KEY`，再回退 `.env.prod` |
| `--images` | 图片目录；按 `--batch-size` 切片，不足则循环复用 |
| `--config` | 压测 JSON（须 `output.type=none`） |
| `--concurrency` | 同时 in-flight job 数（真正的压测旋钮） |
| `--batch-size` | 每发几张，建议先 2 |
| `--duration` | 墙钟秒数 |
| `--timeout` | 单次 submit+SSE 上限秒数 |

跑之前确认：目标环境、config 未开 supabase/volume 保存、Grafana / Modal 面板已打开。跑完看脚本错误分类 + 平台负载，不要只看其中一个。
