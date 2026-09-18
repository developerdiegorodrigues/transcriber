"""Application configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
)
OUTPUT_FORMATS = ("all", "txt", "vtt", "srt", "tsv", "json")


@dataclass(frozen=True, slots=True)
class TranscriptionConfig:
    model: str = "large"
    language: str | None = "English"
    requested_device: str = "auto"
    output_dir: Path = Path("output")
    output_format: str = "all"

    def __post_init__(self) -> None:
        if self.requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Dispositivo inválido: {self.requested_device}")
        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(f"Formato de saída inválido: {self.output_format}")
