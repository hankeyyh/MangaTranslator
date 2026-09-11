"""Modal deployment constants for MangaTranslator (phase 1)."""

APP_NAME = "manga-translator-mt"

MODEL_VOLUME_NAME = "mt-models"
SCRATCH_VOLUME_NAME = "mt-scratch"
JOB_DICT_NAME = "mt-jobs"

MODEL_MOUNT_PATH = "/app/models"
SCRATCH_MOUNT_PATH = "/scratch"
APP_ROOT = "/app"
FONTS_VOLUME_PATH = f"{MODEL_MOUNT_PATH}/fonts"

ENV_SECRET_NAME = "manga-translator-mt-env"

MAX_IMAGES_PER_JOB = 20
WORKER_TIMEOUT_SECONDS = 900  # 15 min, per design doc
GATEWAY_TIMEOUT_SECONDS = 3600
DOWNLOAD_MODELS_TIMEOUT_SECONDS = 3600
SSE_HEARTBEAT_SECONDS = 15
SSE_POLL_SECONDS = 1.0

GATEWAY_CONFIG = {
    "cpu": 1.0,
    "memory": 2048,
    "timeout": GATEWAY_TIMEOUT_SECONDS,
    "min_containers": 0,
    "scaledown_window": 60,
}

GPU_CONFIG = {
    "gpu": "A10G",
    "cpu": 4.0,
    "memory": 16384,
    "timeout": WORKER_TIMEOUT_SECONDS,
    "min_containers": 0,
    "scaledown_window": 300,
}

BASE_IMAGE = "pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime"

APT_PACKAGES = [
    "libsm6",
    "libxext6",
    "libxrender1",
    "libgomp1",
    "libglib2.0-0",
    "libgl1",
    "libegl1",
    "libgles2",
    "libglx-mesa0",
    "libopengl0",
    "curl",
    "wget",
    "git",
    "fonts-dejavu-core",
]

# Gateway: no torch. Worker: torch comes from the CUDA base image.
GATEWAY_PIP_PACKAGES = [
    "fastapi>=0.115",
    "pydantic>=2",
    "python-multipart>=0.0.9",
    "httpx>=0.27",
]

# Keep Gradio / FLUX / SAM out of the production image.
WORKER_PIP_PACKAGES = GATEWAY_PIP_PACKAGES + [
    "deepl>=1.19.0",
    "fonttools>=4.56.0",
    "huggingface_hub>=0.26.0",
    "manga-ocr>=0.1.15",
    "numpy>=1.24.0",
    "opencv-contrib-python-headless>=4.8.0",
    "packaging>=24.0",
    "pillow>=11.1.0",
    "pyoxipng>=9.1.1",
    "pythainlp>=5.3.4",
    "requests>=2.32.3",
    "safetensors>=0.4.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.10.0",
    "skia-python>=87.7",
    "spandrel>=0.3.0",
    "transformers>=4.51.0",
    "uharfbuzz>=0.48.0",
    "ultralytics>=8.3.94",
]

ENV_VARS = {
    "PYTHONPATH": APP_ROOT,
    "PYTHONUNBUFFERED": "1",
    "TORCH_HOME": MODEL_MOUNT_PATH,
    "HF_HOME": f"{MODEL_MOUNT_PATH}/huggingface",
    "HF_HUB_CACHE": f"{MODEL_MOUNT_PATH}/huggingface",
    "TRANSFORMERS_CACHE": f"{MODEL_MOUNT_PATH}/transformers",
    "XDG_CACHE_HOME": f"{MODEL_MOUNT_PATH}/cache",
    "MT_MODELS_DIR": MODEL_MOUNT_PATH,
    "YOLO_CONFIG_DIR": "/tmp",
    "CUDA_VISIBLE_DEVICES": "0",
}

# Individual font files from manga-image-translator/fonts → Volume packs.
MIT_FONT_PACKS = {
    "Anime Ace 3.0": "anime_ace_3.ttf",
    "Comic Shanns 2": "comic shanns 2.ttf",
    "Comic Marker Deluxe": "Comic Marker Deluxe.ttf",
    "Bangers": "Bangers-Regular.ttf",
    "Komika Slim": "KOMIKASL.ttf",
    "Caveat": "Caveat-VariableFont_wght.ttf",
    "Noto Sans": "NotoSans-VariableFont_wdth,wght.ttf",
    "Inter": "Inter-VariableFont_opsz,wght.ttf",
    "Noto Sans SC": "NotoSansSC-VariableFont_wght.ttf",
    "ZCOOL KuaiLe": "ZCOOLKuaiLe-Regular.ttf",
    "Long Cang": "LongCang-Regular.ttf",
    "Ma Shan Zheng": "MaShanZheng-Regular.ttf",
    "Noto Sans JP": "NotoSansJP-VariableFont_wght.ttf",
    "GenEi LateGo N": "GenEiLateGoN_v2.ttf",
    "GenEi Antique": "Genei-Antique.ttf",
    "M PLUS Rounded 1c": "MPLUSRounded1c-Regular.ttf",
    "Zen Kurenaido": "ZenKurenaido-Regular.ttf",
    "Noto Sans KR": "NotoSansKR-VariableFont_wght.ttf",
    "KOMACON": "KOMACON.ttf",
    "Gowun Dodum": "GowunDodum-Regular.ttf",
    "Nanum Pen Script": "NanumPenScript-Regular.ttf",
    "Noto Sans TC": "NotoSansTC-VariableFont_wght.ttf",
    "LXGW WenKai TC": "LXGWWenKaiTC-Regular.ttf",
    "Noto Sans Thai": "NotoSansThai-VariableFont_wdth,wght.ttf",
    "Charmonman": "Charmonman-Regular.ttf",
    "Itim": "Itim-Regular.ttf",
    "Krub": "Krub-Regular.ttf",
    "Playpen Sans Thai": "PlaypenSansThai-VariableFont_wght.ttf",
    "Noto Sans Arabic": "NotoSansArabic-VariableFont_wdth,wght.ttf",
    "QTS Manga": "QTSManga-Regular.ttf",
}

FONT_NAME_ALIASES = {
    "anime ace": "Anime Ace 3.0",
    "anime ace 3": "Anime Ace 3.0",
    "anime ace 3.0": "Anime Ace 3.0",
    "anime-ace-3": "Anime Ace 3.0",
    "komika slim": "Komika Slim",
    "komika-slim": "Komika Slim",
    "noto sans": "Noto Sans",
    "noto-sans": "Noto Sans",
}

PROVIDER_ALIASES = {
    "deepseek": "DeepSeek",
    "deepl": "DeepL",
    "google": "Google",
    "gemini": "Google",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "spacexai": "SpaceXAI",
    "xai": "SpaceXAI",
    "openrouter": "OpenRouter",
}

TERMINAL_EVENTS = frozenset({"job_completed", "job_failed"})
ACTIVE_JOB_STATUSES = frozenset({"queued", "running", "completed"})
