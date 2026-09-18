"""Backend dispatch for a transcription job."""

from __future__ import annotations

from pathlib import Path

from .backends import faster_whisper, openai_whisper
from .config import TranscriptionConfig
from .hardware import probe_cuda, resolve_device
from .result import TranscriptionOutcome
from .separation import separated_vocals


def _dispatch(
    media_path: Path,
    config: TranscriptionConfig,
    device: str,
    *,
    output_media_path: Path | None = None,
) -> TranscriptionOutcome:
    if config.backend == "faster-whisper":
        return faster_whisper.transcribe_file(
            media_path, config, device, output_media_path=output_media_path
        )

    output_dir = openai_whisper.transcribe_file(media_path, config, device)
    output_identity = output_media_path or media_path
    text_path = output_dir / f"{output_identity.stem}.txt"
    text = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else ""
    return TranscriptionOutcome(
        output_dir=output_dir,
        text=text,
        backend=config.backend,
        model=config.model,
        device=device,
    )


def transcribe_file(
    media_path: Path, config: TranscriptionConfig, device: str
) -> TranscriptionOutcome:
    if not config.separate_vocals:
        return _dispatch(media_path, config, device)

    torch_cuda = probe_cuda()
    separation_device = resolve_device(
        config.separation_device, torch_cuda, "openai-whisper"
    )
    with separated_vocals(
        media_path, device=separation_device, model=config.demucs_model
    ) as vocal_path:
        return _dispatch(
            vocal_path,
            config,
            device,
            output_media_path=media_path,
        )
