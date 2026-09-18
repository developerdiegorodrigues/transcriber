"""Backend dispatch for a transcription job."""

from __future__ import annotations

from pathlib import Path

from .backends import faster_whisper, openai_whisper
from .config import TranscriptionConfig
from .result import TranscriptionOutcome


def transcribe_file(
    media_path: Path, config: TranscriptionConfig, device: str
) -> TranscriptionOutcome:
    if config.backend == "faster-whisper":
        return faster_whisper.transcribe_file(media_path, config, device)

    output_dir = openai_whisper.transcribe_file(media_path, config, device)
    text_path = output_dir / f"{media_path.stem}.txt"
    text = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else ""
    return TranscriptionOutcome(
        output_dir=output_dir,
        text=text,
        backend=config.backend,
        model=config.model,
        device=device,
    )
