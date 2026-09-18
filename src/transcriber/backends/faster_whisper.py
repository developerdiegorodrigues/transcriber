"""In-process faster-whisper backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import TranscriptionConfig
from ..errors import DependencyError, TranscriptionError
from ..outputs import write_outputs
from ..result import TranscriptionOutcome


def _language_code(language: str | None) -> str | None:
    if language is None:
        return None
    from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

    normalized = language.casefold()
    if normalized in LANGUAGES:
        return normalized
    return TO_LANGUAGE_CODE.get(normalized, normalized)


def _word_to_dict(word: Any) -> dict[str, Any]:
    return {
        "word": word.word,
        "start": word.start,
        "end": word.end,
        "probability": word.probability,
    }


def _segment_to_dict(segment: Any) -> dict[str, Any]:
    data = {
        "id": segment.id,
        "seek": segment.seek,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "tokens": list(segment.tokens),
        "temperature": segment.temperature,
        "avg_logprob": segment.avg_logprob,
        "compression_ratio": segment.compression_ratio,
        "no_speech_prob": segment.no_speech_prob,
    }
    if segment.words:
        data["words"] = [_word_to_dict(word) for word in segment.words]
    return data


def _batch_candidates(requested: int) -> list[int]:
    candidates: list[int] = []
    current = requested
    while current > 1:
        if current not in candidates:
            candidates.append(current)
        current //= 2
    candidates.append(1)
    return candidates


def _is_out_of_memory(error: RuntimeError) -> bool:
    message = str(error).casefold()
    return any(
        marker in message
        for marker in ("out of memory", "failed to allocate", "cuda_error_out_of_memory")
    )


def _run_inference(
    model: Any,
    media_path: Path,
    config: TranscriptionConfig,
) -> tuple[list[Any], Any, int]:
    from faster_whisper import BatchedInferencePipeline

    common_options = {
        "language": _language_code(config.language),
        "beam_size": config.beam_size,
        "vad_filter": config.vad_filter,
        "word_timestamps": config.word_timestamps,
        "log_progress": True,
    }
    last_error: RuntimeError | None = None
    for batch_size in _batch_candidates(config.batch_size):
        try:
            if batch_size == 1:
                segments, info = model.transcribe(str(media_path), **common_options)
            else:
                pipeline = BatchedInferencePipeline(model=model)
                segments, info = pipeline.transcribe(
                    str(media_path), batch_size=batch_size, **common_options
                )
            return list(segments), info, batch_size
        except RuntimeError as exc:
            if not _is_out_of_memory(exc) or batch_size == 1:
                raise
            last_error = exc
            next_batch = max(1, batch_size // 2)
            print(f"VRAM insuficiente com batch {batch_size}; tentando batch {next_batch}.")
    raise last_error or RuntimeError("Falha inesperada durante a inferência")


def transcribe_file(
    media_path: Path, config: TranscriptionConfig, device: str
) -> TranscriptionOutcome:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise DependencyError(
            "faster-whisper não está instalado. Reinstale usando requirements.lock."
        ) from exc

    compute_type = config.compute_type_for(device)
    output_dir = config.output_dir.expanduser().resolve() / media_path.stem
    print(f"\nTranscrevendo '{media_path.name}'")
    print(
        f"Backend: faster-whisper | modelo: {config.model} | dispositivo: {device} "
        f"| precisão: {compute_type} | batch inicial: {config.batch_size}"
    )
    try:
        model = WhisperModel(config.model, device=device, compute_type=compute_type)
        segments, info, effective_batch = _run_inference(model, media_path, config)
    except Exception as exc:
        if isinstance(exc, (DependencyError, TranscriptionError)):
            raise
        raise TranscriptionError(f"Falha no faster-whisper: {exc}") from exc

    result = {
        "text": "".join(segment.text for segment in segments).strip(),
        "segments": [_segment_to_dict(segment) for segment in segments],
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "duration_after_vad": info.duration_after_vad,
    }
    write_outputs(result, media_path, output_dir, config.output_format)
    print(f"Transcrição concluída: {output_dir} (batch efetivo: {effective_batch})")
    return TranscriptionOutcome(
        output_dir=output_dir,
        text=result["text"],
        backend=config.backend,
        model=config.model,
        device=device,
        compute_type=compute_type,
        batch_size=effective_batch,
    )
