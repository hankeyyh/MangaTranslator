"""Resolve API font_name to a Volume font-pack directory."""

from __future__ import annotations

from pathlib import Path

from deploy.modal_config import FONT_NAME_ALIASES


def _pack_has_fonts(path: Path) -> bool:
    if not path.is_dir():
        return False
    suffixes = {".ttf", ".otf", ".ttc"}
    return any(p.suffix.lower() in suffixes for p in path.iterdir() if p.is_file())


def resolve_font_dir(font_name: str, fonts_root: Path) -> Path:
    if not fonts_root.is_dir():
        raise FileNotFoundError(f"fonts root not found: {fonts_root}")

    requested = (font_name or "").strip() or "Anime Ace 3.0"
    wanted = FONT_NAME_ALIASES.get(requested.lower(), requested)

    direct = fonts_root / wanted
    if _pack_has_fonts(direct):
        return direct

    lowered = wanted.lower()
    for pack in fonts_root.iterdir():
        if pack.is_dir() and pack.name.lower() == lowered and _pack_has_fonts(pack):
            return pack

    for pack in sorted(fonts_root.iterdir()):
        if _pack_has_fonts(pack):
            return pack

    raise FileNotFoundError(
        f"no font pack found for '{font_name}' under {fonts_root}"
    )
