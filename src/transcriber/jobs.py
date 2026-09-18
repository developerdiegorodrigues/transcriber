"""Safe execution and atomic publication of transcription jobs."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import TranscriptionConfig
from .errors import TranscriptionError
from .pipeline import transcribe_file
from .result import TranscriptionOutcome


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _config_metadata(config: TranscriptionConfig) -> dict[str, Any]:
    values = asdict(config)
    values.pop("output_dir", None)
    values["temperatures"] = list(config.temperatures)
    return values


def _write_metadata(
    directory: Path,
    media_path: Path,
    config: TranscriptionConfig,
    outcome: TranscriptionOutcome,
    elapsed_seconds: float,
) -> None:
    artifacts = sorted(
        path.name for path in directory.iterdir() if path.is_file() and not path.name.endswith(".metadata.json")
    )
    report = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "media": {
            "name": media_path.name,
            "size_bytes": media_path.stat().st_size,
        },
        "execution": {
            "elapsed_seconds": round(elapsed_seconds, 3),
            "backend": outcome.backend,
            "model": outcome.model,
            "device": outcome.device,
            "compute_type": outcome.compute_type,
            "batch_size": outcome.batch_size,
        },
        "config": _config_metadata(config),
        "versions": {
            "video-transcriber": _version("video-transcriber"),
            "openai-whisper": _version("openai-whisper"),
            "faster-whisper": _version("faster-whisper"),
            "ctranslate2": _version("ctranslate2"),
            "demucs": _version("demucs"),
        },
        "artifacts": artifacts,
    }
    (directory / f"{media_path.stem}.metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _publish(staged: Path, final: Path, force: bool) -> None:
    if not final.exists():
        staged.replace(final)
        return
    if not force:
        raise TranscriptionError(
            f"A saída já existe: {final}. Use --skip-existing ou --force."
        )

    backup = final.parent / f".{final.name}.backup-{uuid4().hex}"
    final.replace(backup)
    try:
        staged.replace(final)
    except Exception:
        if not final.exists() and backup.exists():
            backup.replace(final)
        raise
    else:
        shutil.rmtree(backup)


def run_job(
    media_path: Path,
    config: TranscriptionConfig,
    device: str,
    *,
    skip_existing: bool = False,
    force: bool = False,
) -> TranscriptionOutcome | None:
    output_root = config.output_dir.expanduser().resolve()
    final = output_root / media_path.stem
    if final.exists():
        if skip_existing:
            print(f"Ignorando '{media_path.name}': saída existente em {final}")
            return None
        if not force:
            raise TranscriptionError(
                f"A saída já existe: {final}. Use --skip-existing ou --force."
            )

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".transcriber-", dir=output_root))
    staged_config = replace(config, output_dir=temporary_root)
    started = time.perf_counter()
    try:
        outcome = transcribe_file(media_path, staged_config, device)
        elapsed = time.perf_counter() - started
        _write_metadata(outcome.output_dir, media_path, config, outcome, elapsed)
        _publish(outcome.output_dir, final, force)
        return replace(outcome, output_dir=final, elapsed_seconds=elapsed)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
