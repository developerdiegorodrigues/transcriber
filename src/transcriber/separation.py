"""Temporary vocal separation for the lyrics profile."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .errors import DependencyError, TranscriptionError


def _extract_audio(media_path: Path, wav_path: Path) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise TranscriptionError("FFmpeg falhou ao preparar o áudio para o Demucs.")


@contextmanager
def separated_vocals(
    media_path: Path, *, device: str, model: str = "htdemucs"
) -> Iterator[Path]:
    try:
        import demucs  # noqa: F401
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise DependencyError("Demucs não está instalado. Reinstale usando requirements.lock.") from exc

    with tempfile.TemporaryDirectory(prefix="transcriber-lyrics-") as temporary_directory:
        root = Path(temporary_directory)
        source = root / "source.wav"
        separated = root / "separated"
        print("Preparando áudio temporário para separação vocal...")
        _extract_audio(media_path, source)
        print(f"Separando vocais com Demucs ({model}, dispositivo: {device})...")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "demucs.separate",
                "--two-stems",
                "vocals",
                "--other-method",
                "none",
                "--name",
                model,
                "--device",
                device,
                "--out",
                str(separated),
                str(source),
            ],
            check=False,
        )
        if result.returncode != 0:
            raise TranscriptionError(
                f"Demucs encerrou com código {result.returncode} durante a separação vocal."
            )
        candidates = list(separated.rglob("vocals.wav"))
        if len(candidates) != 1:
            raise TranscriptionError("Demucs não produziu exatamente um stem vocal.")
        target = root / f"{media_path.stem}.wav"
        candidates[0].replace(target)
        print("Separação concluída; o Demucs foi descarregado antes da transcrição.")
        yield target
