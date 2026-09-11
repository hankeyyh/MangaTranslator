"""
Modal App for MangaTranslator.

Phase 1 public API:
  POST /v1/jobs
  GET  /v1/jobs/{job_id}/events

Does not replace the existing manga-image-translator App (`manga-translator`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
# Local deploy uses the repo root; the container image puts code at /app.
# Do not import sibling `modal_config` — Modal hydrates this file from /root,
# where that module is not on the path.
for _root in (PROJECT_ROOT, Path("/app")):
    if _root.is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from deploy.modal_config import (  # noqa: E402
    APP_NAME,
    APP_ROOT,
    APT_PACKAGES,
    BASE_IMAGE,
    DOWNLOAD_MODELS_TIMEOUT_SECONDS,
    ENV_SECRET_NAME,
    ENV_VARS,
    GATEWAY_CONFIG,
    GATEWAY_PIP_PACKAGES,
    GPU_CONFIG,
    JOB_DICT_NAME,
    MODEL_MOUNT_PATH,
    MODEL_VOLUME_NAME,
    SCRATCH_MOUNT_PATH,
    SCRATCH_VOLUME_NAME,
    WORKER_PIP_PACKAGES,
)

app = modal.App(APP_NAME)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
scratch_volume = modal.Volume.from_name(SCRATCH_VOLUME_NAME, create_if_missing=True)
job_dict = modal.Dict.from_name(JOB_DICT_NAME, create_if_missing=True)
env_secret = modal.Secret.from_name(ENV_SECRET_NAME)
function_secrets = [env_secret]

IGNORE = [
    "**/__pycache__",
    "**/*.pyc",
    "**/.DS_Store",
    "**/test_*.py",
]


def _add_font_sources(image: modal.Image) -> modal.Image:
    mt_fonts = PROJECT_ROOT / "fonts"
    if mt_fonts.is_dir():
        image = image.add_local_dir(str(mt_fonts), "/opt/font-src/mt")
    mit_fonts = PROJECT_ROOT.parent / "manga-image-translator" / "fonts"
    if mit_fonts.is_dir():
        image = image.add_local_dir(str(mit_fonts), "/opt/font-src/mit")
    return image


def _add_service_code(image: modal.Image, include_core: bool) -> modal.Image:
    image = image.add_local_dir(
        str(DEPLOY_DIR), f"{APP_ROOT}/deploy", copy=True, ignore=IGNORE
    )
    if include_core:
        image = image.add_local_dir(
            str(PROJECT_ROOT / "core"), f"{APP_ROOT}/core", ignore=IGNORE
        )
        image = image.add_local_dir(
            str(PROJECT_ROOT / "utils"), f"{APP_ROOT}/utils", ignore=IGNORE
        )
        image = _add_font_sources(image)
    return image


gateway_image = _add_service_code(
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*GATEWAY_PIP_PACKAGES)
    .env({"PYTHONPATH": APP_ROOT, "PYTHONUNBUFFERED": "1"}),
    include_core=False,
)

worker_image = _add_service_code(
    modal.Image.from_registry(BASE_IMAGE)
    .apt_install(*APT_PACKAGES)
    .env(ENV_VARS)
    .pip_install(*WORKER_PIP_PACKAGES),
    include_core=True,
)


@app.function(
    image=worker_image,
    gpu=GPU_CONFIG["gpu"],
    cpu=GPU_CONFIG["cpu"],
    memory=GPU_CONFIG["memory"],
    timeout=GPU_CONFIG["timeout"],
    min_containers=GPU_CONFIG["min_containers"],
    scaledown_window=GPU_CONFIG["scaledown_window"],
    volumes={
        MODEL_MOUNT_PATH: model_volume,
        SCRATCH_MOUNT_PATH: scratch_volume,
    },
    secrets=function_secrets,
)
def process_job(job_id: str) -> None:
    import os
    import sys

    os.chdir(APP_ROOT)
    sys.path.insert(0, APP_ROOT)
    from deploy.api.worker import run_job

    run_job(job_id, job_dict, scratch_volume)


@app.function(
    image=worker_image,
    cpu=4.0,
    memory=16384,
    timeout=DOWNLOAD_MODELS_TIMEOUT_SECONDS,
    volumes={MODEL_MOUNT_PATH: model_volume},
    secrets=function_secrets,
)
def download_models() -> dict:
    import os
    import sys

    os.chdir(APP_ROOT)
    sys.path.insert(0, APP_ROOT)
    from deploy.download_models import download_phase1_models

    result = download_phase1_models()
    model_volume.commit()
    print("volume committed")
    return {"status": "success", **result}


@app.function(
    image=worker_image,
    cpu=1.0,
    memory=2048,
    volumes={
        MODEL_MOUNT_PATH: model_volume,
        SCRATCH_MOUNT_PATH: scratch_volume,
    },
)
def list_volumes() -> dict:
    import os
    from pathlib import Path

    def _walk(root: str) -> list[str]:
        path = Path(root)
        if not path.exists():
            return []
        items = []
        for dirpath, _, filenames in os.walk(path):
            for name in filenames:
                items.append(str(Path(dirpath) / name))
        return items[:200]

    models = _walk(MODEL_MOUNT_PATH)
    scratch = _walk(SCRATCH_MOUNT_PATH)
    print(f"models ({len(models)} files, showing up to 200):")
    for item in models:
        print(f"  {item}")
    print(f"scratch ({len(scratch)} files, showing up to 200):")
    for item in scratch:
        print(f"  {item}")
    return {"models": models, "scratch": scratch}


@app.function(
    image=gateway_image,
    cpu=GATEWAY_CONFIG["cpu"],
    memory=GATEWAY_CONFIG["memory"],
    timeout=GATEWAY_CONFIG["timeout"],
    min_containers=GATEWAY_CONFIG["min_containers"],
    scaledown_window=GATEWAY_CONFIG["scaledown_window"],
    volumes={SCRATCH_MOUNT_PATH: scratch_volume},
    secrets=function_secrets,
)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, APP_ROOT)
    from deploy.api.app import create_app

    return create_app(
        job_backend=job_dict,
        spawn_job=lambda job_id: process_job.spawn(job_id),
        scratch_volume=scratch_volume,
    )


@app.local_entrypoint()
def main() -> None:
    print("MangaTranslator Modal service")
    print("  modal deploy deploy/modal_app.py")
    print("  modal run deploy/modal_app.py::download_models")
    print("  modal run deploy/modal_app.py::list_volumes")
