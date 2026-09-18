"""Shared transcription result types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TranscriptionOutcome:
    output_dir: Path
    text: str
    backend: str
    model: str
    device: str
    compute_type: str | None = None
    batch_size: int | None = None
