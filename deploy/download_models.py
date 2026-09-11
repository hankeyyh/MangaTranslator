"""Download phase-1 models and fonts onto the mt-models Volume."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from deploy.modal_config import FONTS_VOLUME_PATH, MIT_FONT_PACKS, MODEL_MOUNT_PATH

FONT_SRC_MT = Path("/opt/font-src/mt")
FONT_SRC_MIT = Path("/opt/font-src/mit")


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    shutil.copy2(src, dest)
    print(f"copied font {src.name} -> {dest}")


def install_fonts(fonts_root: Path = Path(FONTS_VOLUME_PATH)) -> int:
    fonts_root.mkdir(parents=True, exist_ok=True)
    copied = 0

    if FONT_SRC_MT.is_dir():
        for pack in FONT_SRC_MT.iterdir():
            if not pack.is_dir():
                continue
            dest_dir = fonts_root / pack.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for font in pack.iterdir():
                if font.suffix.lower() in {".ttf", ".otf", ".ttc"}:
                    target = dest_dir / font.name
                    if not target.exists():
                        _copy_file(font, target)
                        copied += 1

    if FONT_SRC_MIT.is_dir():
        for pack_name, filename in MIT_FONT_PACKS.items():
            src = FONT_SRC_MIT / filename
            if not src.is_file():
                print(f"skip missing MIT font: {filename}")
                continue
            dest = fonts_root / pack_name / src.name
            if not dest.exists():
                _copy_file(src, dest)
                copied += 1

    packs = [p.name for p in fonts_root.iterdir() if p.is_dir()] if fonts_root.is_dir() else []
    print(f"font packs available: {packs}")
    return copied


def download_phase1_models(verbose: bool = True) -> dict[str, str]:
    os.chdir("/app")
    os.environ.setdefault("HF_HOME", f"{MODEL_MOUNT_PATH}/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{MODEL_MOUNT_PATH}/transformers")

    from core.ml.model_manager import get_model_manager
    from core.validation import autodetect_yolo_model_path

    models_dir = Path(MODEL_MOUNT_PATH)
    models_dir.mkdir(parents=True, exist_ok=True)
    manager = get_model_manager()

    yolo2 = autodetect_yolo_model_path(models_dir, "yolo_2")
    print(f"ensuring YOLO2 at {yolo2}")
    manager.load_yolo_speech_bubble(str(yolo2), verbose=verbose)

    print("ensuring RT-DETR conjoined bubble model")
    manager.load_rtdetr_conjoined_bubble(verbose=verbose)

    print("ensuring YOLO panel model")
    manager.load_yolo_panel(verbose=verbose)

    print("ensuring YOLO OSB-text (bubble-box verification)")
    manager.load_yolo_osbtext(verbose=verbose)

    print("ensuring LaMa Large inpainting")
    manager.load_lama_large(verbose=verbose)
    manager.unload_lama_large(verbose=verbose)

    print("ensuring DBNet text detector (LaMa stroke mask)")
    manager.load_dbnet(verbose=verbose)
    manager.unload_dbnet(verbose=verbose)

    print("ensuring manga-ocr")
    manager.load_manga_ocr(verbose=verbose)

    copied = install_fonts()
    print(f"fonts copied/new: {copied}")
    return {
        "yolo2": str(yolo2),
        "fonts_root": FONTS_VOLUME_PATH,
    }
