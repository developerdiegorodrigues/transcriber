"""Output writers shared by in-process backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_outputs(
    result: dict[str, Any], media_path: Path, output_dir: Path, output_format: str
) -> None:
    from whisper.utils import get_writer

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = get_writer(output_format, str(output_dir))
    writer(result, str(media_path), options={})
