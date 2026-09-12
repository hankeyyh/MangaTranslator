# MangaTranslator Modal 一期部署

把 `MangaTranslator` 部署成独立 App **`manga-translator-mt`**（不覆盖现网 `manga-translator`）。

对外两个接口：

- `POST /v1/jobs` 批量提交
- `GET /v1/jobs/{job_id}/events` SSE 状态流（支持 `?after=seq`）

网关、Worker、模型预热共用 Secret `manga-translator-mt-env`，由项目根目录 `.env.prod` 创建。需要 `MT_API_KEY`、`DEEPSEEK_API_KEY`、`DEEPL_AUTH_KEY`、`SUPABASE_URL`、`SUPABASE_SERVICE_ROLE_KEY`、`HF_TOKEN`。

```bash
./deploy/deploy.sh setup      # 用 .env.prod 创建 / 覆盖 manga-translator-mt-env
./deploy/deploy.sh deploy     # 部署网关 + GPU Worker（首次构建镜像较慢）
./deploy/deploy.sh models     # 预热 Volume：YOLO2 / RT-DETR / panel / OSB / LaMa Large / DBNet / manga-ocr / 字体（首次deploy需要）
./deploy/deploy.sh test health
./deploy/deploy.sh test fast -i /path/to/page.png
./deploy/deploy.sh test precise -i /path/to/page.png
```

鉴权：`Authorization: Bearer $MT_API_KEY`。

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
