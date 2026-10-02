# MangaTranslator Modal 一期部署

把 `MangaTranslator` 部署成独立 App **`manga-translator-mt`**（不覆盖现网 `manga-translator`）。

对外两个接口：

- `POST /v1/jobs` 批量提交
- `GET /v1/jobs/{job_id}/events` SSE 状态流（支持 `?after=seq`）

网关、Worker、模型预热共用 Secret `manga-translator-mt-env`，由项目根目录 `.env.prod` 创建。需要 `MT_API_KEY`、`DEEPSEEK_API_KEY`、`DEEPL_AUTH_KEY`、`HF_TOKEN`。

```bash
./deploy/deploy.sh setup      # 用 .env.prod 创建 / 覆盖 manga-translator-mt-env
./deploy/deploy.sh deploy     # 部署网关 + GPU Worker（首次构建镜像较慢）
./deploy/deploy.sh models     # 预热 Volume：YOLO2 / RT-DETR / panel / OSB / LaMa Large / DBNet / manga-ocr / 字体（首次deploy需要）
./deploy/deploy.sh test health
./deploy/deploy.sh test fast -i /path/to/page.png
./deploy/deploy.sh test precise -i /path/to/page.png
./deploy/deploy.sh logs          # 实时跟随 App 日志（Ctrl+C 停）
```

鉴权：`Authorization: Bearer $MT_API_KEY`。

## 目录结构

```
deploy/
├── README.md              本说明
├── deploy.sh              部署入口：setup / deploy / models / test / logs
├── modal_app.py           Modal App 定义：web 网关、process_job、download_models、list_volumes
├── modal_config.py        常量：镜像、GPU、Volume 路径、字体别名、默认字体
├── download_models.py     预热 Volume：模型 + 本仓库 fonts/ 字体包
├── smoke_test.py          测试脚本：健康检查 / 提交 job / 消费 SSE
└── api/                   网关与 Worker 实现（随镜像打进 /app/deploy）
    ├── app.py             FastAPI：POST /v1/jobs、GET .../events、拉输出
    ├── schemas.py         请求 / 事件 JSON 模型（JobConfigIn、CreateJobRequest）
    ├── jobs.py            JobStore：Modal Dict 里的 job 元数据与有序事件
    ├── config_map.py      把公开 JobConfig 转成 MangaTranslatorConfig
    ├── fonts.py           font_name → Volume 上的字体包目录
    ├── worker.py          GPU Worker：下载图 → 翻译渲染 → 上传 → 写 SSE 事件
    └── storage.py         输出到 Supabase Storage / Cloudflare R2
```

`modal_app.py` 只负责挂镜像、Volume、Secret 和函数入口；HTTP 与翻译逻辑在 `api/`。字体只来自本仓库 `fonts/`，不依赖隔壁项目。

## SSE 事件

`GET /v1/jobs/{job_id}/events`（可用 `?after=<seq>` 断线续传）。除 `heartbeat` 外，每条都是：

```
id: <seq>
event: <name>
data: <json>

```

`data` 是扁平 JSON：公共字段 `seq` / `event` / `job_id`，再加上该事件自己的字段。`heartbeat` 没有 `id`，`data` 为 `{}`。

| event | data 字段 | 含义 |
|--------|-----------|------|
| `queued` | `position` | 入队（创建 job 时就写） |
| `started` | `worker_id` | Worker 开始处理 |
| `image_progress` | `image_id`, `index`, `stage`, `progress` | 单图阶段；一期只有 `stage=start`、`progress=0.0` |
| `image_completed` | `image_id`, `index`, `output_path?` | 单图成功。`output.type=none` 时无 `output_path` |
| `image_failed` | `image_id`, `index`, `error` | 单图失败；整批继续跑后面的图 |
| `job_completed` | `success_count`, `error_count` | 整批结束（部分图失败也走这个，不走 `job_failed`） |
| `job_failed` | `error` | 循环外的整批失败；job 丢失时网关也会推一条 |
| `heartbeat` | （空对象） | 约每 15s 保活；忽略即可 |

终态是 `job_completed` / `job_failed`，流随后关闭。部分图失败时 job 状态仍是 `completed`，以 `error_count` 区分。

一期默认：OSB 开、擦字 lama_large、不上 FLUX/SAM、fast = DeepSeek one-step、precise = manga-ocr + DeepL two-step。Next.js 不切流。

## 输出保存选项

`output.type` 支持 `none`（默认，不保存）、`volume`、`supabase`、`r2`。
`supabase` 和 `r2` 必须传 `output.bucket`；`output.paths` 可省略，传入时数量必须与图片数量一致。
省略路径时保存为 `<job_id>/<image_id>.<output_format>`。

R2 请求示例：

```json
{
  "images": [{"image_id": "page-1", "url": "https://example.com/page.png", "index": 0}],
  "output": {
    "type": "r2",
    "bucket": "manga-results",
    "paths": ["user/task/result/page-1.webp"]
  }
}
```

在 `.env.prod` 配置以下变量，再运行 `./deploy/deploy.sh setup` 更新 Modal Secret，
运行 `./deploy/deploy.sh deploy` 部署包含 R2 上传依赖的新 Worker：

```dotenv
R2_ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=<R2 S3 Access Key ID>
R2_SECRET_ACCESS_KEY=<R2 S3 Secret Access Key>
```

以上变量分别对应调用方的 `STORAGE_ENDPOINT`、`STORAGE_ACCESS_KEY`、`STORAGE_SECRET_KEY`；
必须使用同一 R2 账户，并确保密钥有目标 Bucket 的写入权限。凭据只配置在服务端，不随 job 请求传递。
实现采用 [Cloudflare 官方 boto3 接入方式](https://developers.cloudflare.com/r2/examples/aws/boto3/)，region 为 `auto`。
选择 Supabase 输出时才需要 `SUPABASE_URL` 和 `SUPABASE_SERVICE_ROLE_KEY`。

R2 输出沿用 Supabase 的图片压缩流程，上传成功后才发出 `image_completed`，
其 `output_path` 为请求中的对象 key（无需公开 Bucket）；上传失败发送 `image_failed`。
