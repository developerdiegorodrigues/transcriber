"""Reproducible performance and accuracy benchmarks."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import config_from_profile
from .errors import DependencyError
from .hardware import probe_cuda, resolve_device
from .pipeline import transcribe_file


def media_duration(path: Path) -> float:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise DependencyError("ffprobe não encontrado; ele normalmente acompanha o FFmpeg.")
    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise DependencyError(f"Não foi possível obter a duração de {path.name}.")
    return float(result.stdout.strip())


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = _words(reference)
    actual = _words(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for row, expected_word in enumerate(expected, start=1):
        current = [row]
        for column, actual_word in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_word != actual_word),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def _prepare_model(backend: str, model: str) -> None:
    if backend == "faster-whisper":
        from faster_whisper.utils import download_model

        print(f"Preparando modelo '{model}' fora da medição...")
        download_model(model)


class GpuMemoryMonitor(AbstractContextManager["GpuMemoryMonitor"]):
    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuMemoryMonitor":
        if shutil.which("nvidia-smi"):
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _sample(self) -> None:
        process_id = str(os.getpid())
        while not self._stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) == 2 and fields[0] == process_id and fields[1].isdigit():
                        memory = int(fields[1])
                        self.peak_mib = max(self.peak_mib or 0, memory)
            self._stop.wait(self.interval)


def run_benchmark(
    media_path: Path,
    profiles: list[str],
    output_path: Path,
    *,
    language: str | None = "English",
    reference: str | None = None,
) -> dict[str, Any]:
    duration = media_duration(media_path)
    cuda = probe_cuda()
    results: list[dict[str, Any]] = []
    artifacts_root = output_path.parent / "artifacts" / media_path.stem

    for profile_name in profiles:
        config = config_from_profile(
            profile_name,
            language=language,
            output_dir=artifacts_root / profile_name,
            output_format="all",
        )
        device = resolve_device(config.requested_device, cuda, config.backend)
        _prepare_model(config.backend, config.model)
        started = time.perf_counter()
        with GpuMemoryMonitor() as memory:
            outcome = transcribe_file(media_path, config, device)
        elapsed = time.perf_counter() - started
        result: dict[str, Any] = {
            "profile": profile_name,
            "backend": outcome.backend,
            "model": outcome.model,
            "device": outcome.device,
            "compute_type": outcome.compute_type,
            "batch_size": outcome.batch_size,
            "elapsed_seconds": round(elapsed, 3),
            "audio_seconds": round(duration, 3),
            "realtime_factor": round(elapsed / duration, 4) if duration else None,
            "audio_x_realtime": round(duration / elapsed, 2) if elapsed else None,
            "peak_gpu_memory_mib": memory.peak_mib,
            "output_dir": str(outcome.output_dir),
        }
        if reference is not None:
            result["word_error_rate"] = round(word_error_rate(reference, outcome.text), 6)
        results.append(result)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "media": str(media_path),
        "system": platform.platform(),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
