"""Application configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
)
OUTPUT_FORMATS = ("all", "txt", "vtt", "srt", "tsv", "json")
BACKENDS = ("faster-whisper", "openai-whisper")


@dataclass(frozen=True, slots=True)
class Profile:
    backend: str
    model: str
    requested_device: str
    compute_type: str | None
    batch_size: int
    beam_size: int
    vad_filter: bool
    word_timestamps: bool = False
    separate_vocals: bool = False
    quality_review: bool = False
    retry_low_confidence: bool = False
    condition_on_previous_text: bool = True
    temperatures: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    hallucination_silence_threshold: float | None = None


PROFILES: dict[str, Profile] = {
    "fast": Profile("faster-whisper", "turbo", "auto", None, 8, 5, True),
    "quality": Profile("faster-whisper", "large-v3", "auto", None, 4, 5, False),
    "cpu": Profile("faster-whisper", "small", "cpu", "int8", 4, 5, True),
    "legacy": Profile("openai-whisper", "large", "auto", None, 1, 5, False),
    "lyrics": Profile(
        "faster-whisper",
        "large-v3",
        "auto",
        None,
        4,
        5,
        False,
        True,
        separate_vocals=True,
        quality_review=True,
        retry_low_confidence=True,
        condition_on_previous_text=False,
        temperatures=(0.0, 0.2, 0.4),
        hallucination_silence_threshold=2.0,
    ),
}


@dataclass(frozen=True, slots=True)
class TranscriptionConfig:
    backend: str = "faster-whisper"
    model: str = "turbo"
    language: str | None = "English"
    requested_device: str = "auto"
    output_dir: Path = Path("output")
    output_format: str = "all"
    compute_type: str | None = None
    batch_size: int = 8
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = False
    separate_vocals: bool = False
    separation_device: str = "auto"
    demucs_model: str = "htdemucs"
    quality_review: bool = False
    retry_low_confidence: bool = False
    max_retry_ranges: int = 3
    condition_on_previous_text: bool = True
    temperatures: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    hallucination_silence_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Dispositivo inválido: {self.requested_device}")
        if self.backend not in BACKENDS:
            raise ValueError(f"Backend inválido: {self.backend}")
        if self.separation_device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Dispositivo de separação inválido: {self.separation_device}")
        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(f"Formato de saída inválido: {self.output_format}")
        if self.batch_size < 1:
            raise ValueError("O batch size deve ser maior que zero")
        if self.beam_size < 1:
            raise ValueError("O beam size deve ser maior que zero")
        if self.max_retry_ranges < 0:
            raise ValueError("O limite de novas tentativas não pode ser negativo")
        if self.separate_vocals and self.backend != "faster-whisper":
            raise ValueError("A separação vocal requer o backend faster-whisper")

    def compute_type_for(self, device: str) -> str:
        if self.compute_type:
            return self.compute_type
        return "float16" if device == "cuda" else "int8"


def config_from_profile(
    profile_name: str,
    *,
    backend: str | None = None,
    model: str | None = None,
    language: str | None = "English",
    requested_device: str | None = None,
    output_dir: Path = Path("output"),
    output_format: str = "all",
    compute_type: str | None = None,
    batch_size: int | None = None,
    beam_size: int | None = None,
    vad_filter: bool | None = None,
    word_timestamps: bool | None = None,
    separate_vocals: bool | None = None,
    separation_device: str = "auto",
    demucs_model: str = "htdemucs",
    quality_review: bool | None = None,
    retry_low_confidence: bool | None = None,
) -> TranscriptionConfig:
    try:
        profile = PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(f"Perfil desconhecido: {profile_name}") from exc
    return TranscriptionConfig(
        backend=backend or profile.backend,
        model=model or profile.model,
        language=language,
        requested_device=requested_device or profile.requested_device,
        output_dir=output_dir,
        output_format=output_format,
        compute_type=compute_type or profile.compute_type,
        batch_size=batch_size or profile.batch_size,
        beam_size=beam_size or profile.beam_size,
        vad_filter=profile.vad_filter if vad_filter is None else vad_filter,
        word_timestamps=(
            profile.word_timestamps if word_timestamps is None else word_timestamps
        ),
        separate_vocals=(
            profile.separate_vocals if separate_vocals is None else separate_vocals
        ),
        separation_device=separation_device,
        demucs_model=demucs_model,
        quality_review=(
            profile.quality_review if quality_review is None else quality_review
        ),
        retry_low_confidence=(
            profile.retry_low_confidence
            if retry_low_confidence is None
            else retry_low_confidence
        ),
        condition_on_previous_text=profile.condition_on_previous_text,
        temperatures=profile.temperatures,
        hallucination_silence_threshold=profile.hallucination_silence_threshold,
    )
