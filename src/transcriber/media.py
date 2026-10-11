"""Media discovery helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import VIDEO_EXTENSIONS
from .errors import DependencyError


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


def media_duration(path: Path) -> float:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise DependencyError("ffprobe não encontrado; ele normalmente acompanha o FFmpeg.")
    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise DependencyError(f"Não foi possível obter a duração de {path.name}.")
    return float(result.stdout.strip())
