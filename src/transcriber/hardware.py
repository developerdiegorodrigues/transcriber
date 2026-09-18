"""Runtime and CUDA diagnostics."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from .errors import HardwareError


@dataclass(frozen=True, slots=True)
class CudaStatus:
    torch_version: str | None
    runtime_version: str | None
    available: bool
    devices: tuple[str, ...] = ()
    total_memory_mib: tuple[int, ...] = ()
    error: str | None = None


def probe_cuda() -> CudaStatus:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on the local installation
        return CudaStatus(None, None, False, error=str(exc))

    try:
        available = bool(torch.cuda.is_available())
        devices: list[str] = []
        memory: list[int] = []
        if available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append(properties.name)
                memory.append(round(properties.total_memory / (1024 * 1024)))
        return CudaStatus(
            torch_version=torch.__version__,
            runtime_version=torch.version.cuda,
            available=available,
            devices=tuple(devices),
            total_memory_mib=tuple(memory),
        )
    except Exception as exc:  # pragma: no cover - hardware/driver dependent
        return CudaStatus(
            torch_version=getattr(torch, "__version__", None),
            runtime_version=getattr(torch.version, "cuda", None),
            available=False,
            error=str(exc),
        )


def _ctranslate2_cuda_available() -> tuple[bool, str | None]:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0, None
    except Exception as exc:  # pragma: no cover - installation/driver dependent
        return False, str(exc)


def resolve_device(
    requested: str,
    status: CudaStatus | None = None,
    backend: str = "openai-whisper",
) -> str:
    status = status or probe_cuda()
    available = status.available
    error = status.error
    if backend == "faster-whisper":
        available, error = _ctranslate2_cuda_available()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not available:
            detail = f" ({error})" if error else ""
            raise HardwareError(
                f"CUDA foi solicitado, mas o backend {backend} não consegue acessar a GPU"
                f"{detail}. Execute 'transcriber doctor' para diagnosticar."
            )
        return "cuda"
    if requested == "auto":
        return "cuda" if available else "cpu"
    raise HardwareError(f"Dispositivo desconhecido: {requested}")


def _package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "não instalado"


def _whisper_executable() -> str:
    beside_python = Path(sys.executable).parent / "whisper"
    if beside_python.is_file():
        return str(beside_python)
    return shutil.which("whisper") or "não encontrado"


def _nvidia_smi_status() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return "não encontrado"
    result = subprocess.run(
        [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip() or "executou sem retornar dispositivos"
    error = (result.stderr.strip() or result.stdout.strip()).splitlines()
    return f"falhou ({error[0] if error else f'código {result.returncode}'})"


def diagnostic_report() -> dict[str, Any]:
    """Return a machine-readable report without user-specific filesystem paths."""

    cuda = probe_cuda()
    ct2_cuda, ct2_error = _ctranslate2_cuda_available()
    devices = [
        {
            "index": index,
            "name": name,
            "total_memory_mib": cuda.total_memory_mib[index],
        }
        for index, name in enumerate(cuda.devices)
    ]
    return {
        "schema_version": 1,
        "system": {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
        },
        "dependencies": {
            "openai-whisper": _package_version("openai-whisper"),
            "faster-whisper": _package_version("faster-whisper"),
            "ctranslate2": _package_version("ctranslate2"),
            "demucs": _package_version("demucs"),
            "ffmpeg_available": shutil.which("ffmpeg") is not None,
            "whisper_cli_available": _whisper_executable() != "não encontrado",
            "nvidia_smi": _nvidia_smi_status(),
        },
        "cuda": {
            "pytorch": {
                "available": cuda.available,
                "torch_version": cuda.torch_version,
                "runtime_version": cuda.runtime_version,
                "error": cuda.error,
            },
            "ctranslate2": {
                "available": ct2_cuda,
                "error": ct2_error,
            },
            "devices": devices,
        },
        "backends": {
            "openai-whisper": {
                "cuda_available": cuda.available,
                "automatic_device": "cuda" if cuda.available else "cpu",
            },
            "faster-whisper": {
                "cuda_available": ct2_cuda,
                "automatic_device": "cuda" if ct2_cuda else "cpu",
            },
        },
    }


def diagnostic_lines(report: dict[str, Any] | None = None) -> list[str]:
    report = report or diagnostic_report()
    system = report["system"]
    dependencies = report["dependencies"]
    cuda = report["cuda"]
    pytorch = cuda["pytorch"]
    ctranslate2 = cuda["ctranslate2"]
    backends = report["backends"]
    lines = [
        "Diagnóstico do transcriber",
        f"Sistema: {system['platform']}",
        f"Python: {system['python_version']}",
        f"openai-whisper: {dependencies['openai-whisper']}",
        f"Executável whisper: {'encontrado' if dependencies['whisper_cli_available'] else 'não encontrado'}",
        f"ffmpeg: {'encontrado' if dependencies['ffmpeg_available'] else 'não encontrado'}",
        f"nvidia-smi: {dependencies['nvidia_smi']}",
        f"PyTorch: {pytorch['torch_version'] or 'não instalado'}",
        f"CUDA do PyTorch: {pytorch['runtime_version'] or 'indisponível'}",
        f"CUDA disponível: {'sim' if pytorch['available'] else 'não'}",
        f"faster-whisper: {dependencies['faster-whisper']}",
        f"CTranslate2: {dependencies['ctranslate2']}",
        f"CUDA no CTranslate2: {'sim' if ctranslate2['available'] else 'não'}",
        f"Demucs: {dependencies['demucs']}",
    ]
    if pytorch["error"]:
        lines.append(f"Erro CUDA: {pytorch['error']}")
    if ctranslate2["error"]:
        lines.append(f"Erro CTranslate2/CUDA: {ctranslate2['error']}")
    for device in cuda["devices"]:
        lines.append(
            f"GPU {device['index']}: {device['name']} "
            f"({device['total_memory_mib']} MiB)"
        )
    lines.append(
        f"Dispositivo automático (openai-whisper): "
        f"{backends['openai-whisper']['automatic_device']}"
    )
    lines.append(
        f"Dispositivo automático (faster-whisper): "
        f"{backends['faster-whisper']['automatic_device']}"
    )
    return lines
