"""Media discovery helpers."""

from __future__ import annotations

from pathlib import Path

from .config import VIDEO_EXTENSIONS


def list_video_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )


def validate_video_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Arquivo não encontrado: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(f"Formato não suportado: {path.suffix or '(sem extensão)'}. Use: {supported}")
    return path
