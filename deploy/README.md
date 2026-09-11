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

一期默认：OSB 开、擦字 lama_large、不上 FLUX/SAM、fast = DeepSeek one-step、precise = manga-ocr + DeepL two-step。Next.js 不切流。
