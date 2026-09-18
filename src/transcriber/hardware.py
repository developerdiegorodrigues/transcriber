"""Runtime and CUDA diagnostics."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

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


def diagnostic_lines() -> list[str]:
    cuda = probe_cuda()
    ct2_version = _package_version("ctranslate2")
    ct2_cuda, ct2_error = _ctranslate2_cuda_available()
    lines = [
        "Diagnóstico do transcriber",
        f"Sistema: {platform.platform()}",
        f"Python: {sys.version.split()[0]} ({sys.executable})",
        f"openai-whisper: {_package_version('openai-whisper')}",
        f"Executável whisper: {_whisper_executable()}",
        f"ffmpeg: {shutil.which('ffmpeg') or 'não encontrado'}",
        f"nvidia-smi: {_nvidia_smi_status()}",
        f"PyTorch: {cuda.torch_version or 'não instalado'}",
        f"CUDA do PyTorch: {cuda.runtime_version or 'indisponível'}",
        f"CUDA disponível: {'sim' if cuda.available else 'não'}",
        f"faster-whisper: {_package_version('faster-whisper')}",
        f"CTranslate2: {ct2_version}",
        f"CUDA no CTranslate2: {'sim' if ct2_cuda else 'não'}",
        f"Demucs: {_package_version('demucs')}",
    ]
    if cuda.error:
        lines.append(f"Erro CUDA: {cuda.error}")
    if ct2_error:
        lines.append(f"Erro CTranslate2/CUDA: {ct2_error}")
    for index, name in enumerate(cuda.devices):
        memory = cuda.total_memory_mib[index]
        lines.append(f"GPU {index}: {name} ({memory} MiB)")
    lines.append(
        f"Dispositivo automático (openai-whisper): "
        f"{resolve_device('auto', cuda, 'openai-whisper')}"
    )
    lines.append(
        f"Dispositivo automático (faster-whisper): "
        f"{resolve_device('auto', cuda, 'faster-whisper')}"
    )
    return lines
