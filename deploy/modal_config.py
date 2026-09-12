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
    "max_inputs": 2,
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

DEFAULT_FONT_NAME = "NotoSans"

FONT_NAME_ALIASES = {
    "noto_sans": "NotoSans",
    "komika_slim": "komika_slim",
    "arial_unicode": "Arial-Unicode",
    "cc_wild_words": "CC-Wild-Words",
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
