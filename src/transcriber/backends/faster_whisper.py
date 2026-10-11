"""In-process faster-whisper backend."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import TranscriptionConfig
from ..cuda_runtime import preload_cuda_libraries
from ..errors import DependencyError, TranscriptionError
from ..hardware import GpuMemory, gpu_memory, supported_compute_types
from ..media import media_duration
from ..outputs import write_outputs
from ..quality import (
    annotate_quality,
    average_log_probability,
    retry_ranges,
    write_quality_report,
)
from ..result import TranscriptionOutcome
from ..vram import BatchPlan, describe_allocation_failure, plan_batch_size


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


#: Whisper's own analysis window. Batching without VAD slices the audio into windows
#: of this length, because the batched pipeline needs segment boundaries.
WINDOW_SECONDS = 30.0


def _window_clips(duration: float, window: float = WINDOW_SECONDS) -> list[dict[str, float]]:
    """Cover the whole audio with consecutive windows.

    Each clip given to the batched pipeline is transcribed up to 30 seconds only, so
    the windows must be at most that long or audio would be dropped silently.
    """

    clips: list[dict[str, float]] = []
    start = 0.0
    while start < duration:
        clips.append({"start": start, "end": min(start + window, duration)})
        start += window
    return clips


def _requires_segment_boundaries(error: Exception) -> bool:
    return "clip timestamps" in str(error).casefold()


def _is_out_of_memory(error: RuntimeError) -> bool:
    message = str(error).casefold()
    return any(
        marker in message
        for marker in ("out of memory", "failed to allocate", "cuda_error_out_of_memory")
    )


def _batched_clips(
    media_path: Path, config: TranscriptionConfig
) -> list[dict[str, float]] | None:
    """Return the windows that let the batched pipeline run without VAD.

    ``None`` means batching is not possible for this configuration: the batched
    pipeline derives its chunks from VAD and refuses any audio longer than one window
    without explicit boundaries.
    """

    if config.vad_filter:
        return None
    if not config.window_batching:
        return None
    return _window_clips(media_duration(media_path))


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
        # The batched pipeline defaults to dropping timestamp tokens, which collapses
        # each chunk into one subtitle block spanning the whole window. Asking for them
        # explicitly keeps subtitles usable and costs nothing measurable.
        "without_timestamps": False,
        "log_progress": True,
        "temperature": config.temperatures,
        "condition_on_previous_text": config.condition_on_previous_text,
        "hallucination_silence_threshold": config.hallucination_silence_threshold,
    }
    clips = _batched_clips(media_path, config)
    if not config.vad_filter and clips is None and config.batch_size > 1:
        print(
            "Inferência em lote exige fronteiras de segmento, que vêm do VAD; "
            "com o VAD desligado, este perfil usa batch 1. "
            "Use --window-batching para processar em lote por janelas de 30s."
        )
        segments, info = model.transcribe(str(media_path), **common_options)
        return list(segments), info, 1

    last_error: RuntimeError | None = None
    for batch_size in _batch_candidates(config.batch_size):
        try:
            if batch_size == 1:
                segments, info = model.transcribe(str(media_path), **common_options)
            else:
                pipeline = BatchedInferencePipeline(model=model)
                segments, info = pipeline.transcribe(
                    str(media_path),
                    batch_size=batch_size,
                    clip_timestamps=clips,
                    **common_options,
                )
            return list(segments), info, batch_size
        except RuntimeError as exc:
            if _requires_segment_boundaries(exc):
                print(
                    "O backend recusou a inferência em lote sem fronteiras de "
                    "segmento; repetindo com batch 1."
                )
                segments, info = model.transcribe(str(media_path), **common_options)
                return list(segments), info, 1
            if not _is_out_of_memory(exc) or batch_size == 1:
                raise
            last_error = exc
            next_batch = max(1, batch_size // 2)
            print(f"VRAM insuficiente com batch {batch_size}; tentando batch {next_batch}.")
    raise last_error or RuntimeError("Falha inesperada durante a inferência")


def _retry_low_confidence_ranges(
    model: Any,
    media_path: Path,
    segments: list[Any],
    config: TranscriptionConfig,
) -> tuple[list[Any], int, int]:
    ranges = retry_ranges(segments, config.max_retry_ranges)
    if not ranges:
        return segments, 0, 0

    replacements: dict[int, list[Any]] = {}
    consumed: set[int] = set()
    replaced = 0
    for retry_range in ranges:
        print(
            f"Reavaliando trecho de baixa confiança: "
            f"{retry_range.start:.2f}s–{retry_range.end:.2f}s"
        )
        retried, _ = model.transcribe(
            str(media_path),
            language=_language_code(config.language),
            beam_size=config.beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=config.word_timestamps,
            clip_timestamps=[retry_range.start, retry_range.end],
            log_progress=False,
        )
        replacement = list(retried)
        original = [segments[index] for index in retry_range.indices]
        if replacement and average_log_probability(replacement) > average_log_probability(original):
            replacements[retry_range.indices[0]] = replacement
            consumed.update(retry_range.indices)
            replaced += 1

    merged: list[Any] = []
    for index, segment in enumerate(segments):
        if index in replacements:
            merged.extend(replacements[index])
        elif index not in consumed:
            merged.append(segment)
    return merged, len(ranges), replaced


def _plan_batch(
    config: TranscriptionConfig, compute_type: str, memory: GpuMemory | None
) -> BatchPlan:
    return plan_batch_size(
        config.batch_size,
        free_mib=memory.free_mib if memory is not None else None,
        model=config.model,
        compute_type=compute_type,
        explicit=config.batch_size_explicit,
    )


def _load_model(
    model: str, device: str, compute_type: str, memory: GpuMemory | None
) -> Any:
    """Load the model, turning an allocation failure into an actionable message."""

    from faster_whisper import WhisperModel

    try:
        return WhisperModel(model, device=device, compute_type=compute_type)
    except ValueError as exc:
        if "compute type" not in str(exc).casefold():
            raise
        supported = supported_compute_types(device)
        detail = f" Disponíveis em {device}: {', '.join(supported)}." if supported else ""
        raise TranscriptionError(
            f"Precisão '{compute_type}' não é suportada por este dispositivo.{detail}"
        ) from exc
    except RuntimeError as exc:
        if device != "cuda" or not _is_out_of_memory(exc):
            raise
        raise TranscriptionError(
            describe_allocation_failure(
                free_mib=memory.free_mib if memory is not None else None,
                model=model,
                compute_type=compute_type,
                device_total_mib=memory.total_mib if memory is not None else None,
            )
        ) from exc


def transcribe_file(
    media_path: Path,
    config: TranscriptionConfig,
    device: str,
    *,
    output_media_path: Path | None = None,
) -> TranscriptionOutcome:
    if device == "cuda":
        libraries = preload_cuda_libraries()
        if not libraries.complete:
            raise DependencyError(
                f"{libraries.describe_missing()}. "
                "Execute 'transcriber doctor' para diagnosticar."
            )

    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise DependencyError(
            "faster-whisper não está instalado. Reinstale usando requirements.lock."
        ) from exc

    compute_type = config.compute_type_for(device)
    output_media_path = output_media_path or media_path
    output_dir = config.output_dir.expanduser().resolve() / output_media_path.stem
    memory = gpu_memory() if device == "cuda" else None
    plan = _plan_batch(config, compute_type, memory)
    print(f"\nTranscrevendo '{media_path.name}'")
    print(
        f"Backend: faster-whisper | modelo: {config.model} | dispositivo: {device} "
        f"| precisão: {compute_type} | batch inicial: {plan.batch_size}"
    )
    if memory is not None:
        print(f"VRAM: {memory.free_mib} MiB livres de {memory.total_mib} MiB | {plan.reason}")
    if not plan.fits:
        print(
            "Aviso: a VRAM livre pode não acomodar este modelo; "
            "se falhar, a mensagem de erro trará as alternativas."
        )
    staged = replace(config, batch_size=plan.batch_size)
    try:
        model = _load_model(config.model, device, compute_type, memory)
        segments, info, effective_batch = _run_inference(model, media_path, staged)
        attempted_retries = 0
        accepted_retries = 0
        if config.retry_low_confidence:
            segments, attempted_retries, accepted_retries = _retry_low_confidence_ranges(
                model, media_path, segments, staged
            )
    except Exception as exc:
        if isinstance(exc, (DependencyError, TranscriptionError)):
            raise
        raise TranscriptionError(f"Falha no faster-whisper: {exc}") from exc

    segment_data = [_segment_to_dict(segment) for segment in segments]
    for index, segment in enumerate(segment_data):
        segment["id"] = index
    result: dict[str, Any] = {
        "text": "".join(segment.text for segment in segments).strip(),
        "segments": segment_data,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "duration_after_vad": info.duration_after_vad,
    }
    if config.quality_review:
        summary = annotate_quality(segment_data)
        summary.update(
            {
                "retry_ranges_attempted": attempted_retries,
                "retry_ranges_accepted": accepted_retries,
            }
        )
        result["quality_summary"] = summary
    write_outputs(result, output_media_path, output_dir, config.output_format)
    if config.quality_review:
        write_quality_report(output_dir, output_media_path.stem, segment_data, summary)
    print(f"Transcrição concluída: {output_dir} (batch efetivo: {effective_batch})")
    return TranscriptionOutcome(
        output_dir=output_dir,
        text=result["text"],
        backend=config.backend,
        model=config.model,
        device=device,
        compute_type=compute_type,
        batch_size=effective_batch,
        requested_batch_size=plan.batch_size,
        free_vram_mib=memory.free_mib if memory is not None else None,
    )
