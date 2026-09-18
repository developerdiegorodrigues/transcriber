"""Adapter for the OpenAI Whisper command-line interface."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from ..config import TranscriptionConfig
from ..errors import DependencyError, TranscriptionError


def find_whisper_executable() -> Path:
    beside_python = Path(sys.executable).parent / "whisper"
    if beside_python.is_file():
        return beside_python
    executable = shutil.which("whisper")
    if executable:
        return Path(executable)
    raise DependencyError(
        "Whisper não está instalado neste ambiente. Instale as dependências descritas no README."
    )


def build_command(
    media_path: Path,
    output_dir: Path,
    config: TranscriptionConfig,
    device: str,
    executable: Path | None = None,
) -> list[str]:
    command = [
        str(executable or find_whisper_executable()),
        str(media_path),
        "--model",
        config.model,
        "--device",
        device,
        "--output_dir",
        str(output_dir),
        "--output_format",
        config.output_format,
        "--fp16",
        "True" if device == "cuda" else "False",
    ]
    if config.language:
        command.extend(["--language", config.language])
    return command


def transcribe_file(media_path: Path, config: TranscriptionConfig, device: str) -> Path:
    output_dir = config.output_dir.expanduser().resolve() / media_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(media_path, output_dir, config, device)
    print(f"\nTranscrevendo '{media_path.name}'")
    print(f"Modelo: {config.model} | dispositivo: {device} | saída: {output_dir}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise TranscriptionError(
            f"O Whisper encerrou com código {result.returncode} ao processar {media_path.name}."
        )
    print(f"Transcrição concluída: {output_dir}")
    return output_dir
