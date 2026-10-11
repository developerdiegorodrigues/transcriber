"""Runtime and CUDA diagnostics."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from .cuda_runtime import CudaLibraryStatus, preload_cuda_libraries
from .errors import HardwareError


@dataclass(frozen=True, slots=True)
class CudaStatus:
    torch_version: str | None
    runtime_version: str | None
    available: bool
    devices: tuple[str, ...] = ()
    total_memory_mib: tuple[int, ...] = ()
    error: str | None = None
    capabilities: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()


def probe_cuda() -> CudaStatus:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on the local installation
        return CudaStatus(None, None, False, error=str(exc))

    try:
        available = bool(torch.cuda.is_available())
        devices: list[str] = []
        memory: list[int] = []
        capabilities: list[str] = []
        architectures: list[str] = []
        if available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append(properties.name)
                memory.append(round(properties.total_memory / (1024 * 1024)))
                capabilities.append(f"sm_{properties.major}{properties.minor}")
            architectures = list(torch.cuda.get_arch_list())
        return CudaStatus(
            torch_version=torch.__version__,
            runtime_version=torch.version.cuda,
            available=available,
            devices=tuple(devices),
            total_memory_mib=tuple(memory),
            capabilities=tuple(capabilities),
            architectures=tuple(architectures),
        )
    except Exception as exc:  # pragma: no cover - hardware/driver dependent
        return CudaStatus(
            torch_version=getattr(torch, "__version__", None),
            runtime_version=getattr(torch.version, "cuda", None),
            available=False,
            error=str(exc),
        )


def supported_compute_types(device: str) -> list[str]:
    """List the precisions CTranslate2 can actually use on a device.

    The answer is architecture dependent: INT8 and BF16 availability on a GPU changes
    with the compute capability, so this is reported instead of assumed.
    """

    try:
        import ctranslate2

        return sorted(ctranslate2.get_supported_compute_types(device))
    except Exception:  # pragma: no cover - installation/driver dependent
        return []


def _supported_compute_types() -> dict[str, list[str]]:
    return {device: supported_compute_types(device) for device in ("cpu", "cuda")}


def _ctranslate2_driver_visible() -> tuple[bool, str | None]:
    """Report whether CTranslate2 sees a CUDA device through the driver API.

    This probe never loads cuBLAS or cuDNN, so it succeeds on an environment where
    inference would still fail. Use :func:`_ctranslate2_cuda_available` to decide
    whether the GPU is actually usable.
    """

    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0, None
    except Exception as exc:  # pragma: no cover - installation/driver dependent
        return False, str(exc)


def _ctranslate2_cuda_available() -> tuple[bool, str | None]:
    """Report whether CTranslate2 can really run on the GPU.

    Requires both a visible device and the CUDA libraries the wheel links against.
    """

    visible, error = _ctranslate2_driver_visible()
    if not visible:
        return False, error
    libraries = preload_cuda_libraries()
    if not libraries.complete:
        return False, libraries.describe_missing()
    return True, None


def cuda_smoke_test(model: str = "tiny") -> tuple[bool, str | None]:
    """Run a minimal GPU inference to prove the CUDA stack works end to end.

    Loading a model and transcribing one second of synthetic audio is the only honest
    way to confirm that cuBLAS and cuDNN kernels execute on this architecture.
    """

    available, error = _ctranslate2_cuda_available()
    if not available:
        return False, error
    try:
        import numpy

        from faster_whisper import WhisperModel

        audio = (
            0.05
            * numpy.sin(2 * numpy.pi * 440 * numpy.arange(16000, dtype=numpy.float32) / 16000)
        ).astype(numpy.float32)
        engine = WhisperModel(model, device="cuda", compute_type="float16")
        segments, _ = engine.transcribe(audio, language="en", beam_size=1, vad_filter=False)
        list(segments)
    except Exception as exc:  # pragma: no cover - hardware/driver dependent
        return False, str(exc)
    return True, None


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


def _nvidia_query(fields: str, index: int = 0) -> list[str] | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    result = subprocess.run(
        [executable, f"--id={index}", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or not lines:
        return None
    return [field.strip() for field in lines[0].split(",")]


@dataclass(frozen=True, slots=True)
class GpuMemory:
    free_mib: int
    total_mib: int


def gpu_memory(index: int = 0) -> GpuMemory | None:
    """Query device-wide free VRAM without creating a CUDA context.

    ``nvidia-smi`` is used on purpose: ``torch.cuda.mem_get_info`` would initialise
    CUDA in this process just to answer the question, and the number has to include
    memory held by *other* processes anyway.
    """

    fields = _nvidia_query("memory.free,memory.total", index)
    if fields is None or len(fields) != 2 or not all(f.isdigit() for f in fields):
        return None
    return GpuMemory(free_mib=int(fields[0]), total_mib=int(fields[1]))


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


def concurrent_compute_usage() -> dict[str, int] | None:
    """Summarise GPU compute processes other than this one.

    Only counts and totals are reported: process names are filesystem paths and the
    reports are meant to be shareable.
    """

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    own = str(os.getpid())
    processes = 0
    memory_mib = 0
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[1].isdigit() or fields[0] == own:
            continue
        processes += 1
        memory_mib += int(fields[1])
    return {"processes": processes, "memory_mib": memory_mib}


def hardware_snapshot() -> dict[str, Any]:
    """Describe the machine well enough to compare measurements across GPUs."""

    cuda = probe_cuda()
    memory = gpu_memory()
    driver = _nvidia_query("driver_version")
    return {
        "gpu": cuda.devices[0] if cuda.devices else None,
        "compute_capability": cuda.capabilities[0] if cuda.capabilities else None,
        "driver_version": driver[0] if driver else None,
        "total_memory_mib": memory.total_mib if memory else None,
        "free_memory_mib_at_start": memory.free_mib if memory else None,
        "architectures": list(cuda.architectures),
        "torch_version": cuda.torch_version,
        "torch_cuda_runtime": cuda.runtime_version,
        "ctranslate2": _package_version("ctranslate2"),
        "faster-whisper": _package_version("faster-whisper"),
        "supported_compute_types": _supported_compute_types(),
        "concurrent_compute": concurrent_compute_usage(),
    }


def _libraries_report(libraries: CudaLibraryStatus) -> dict[str, Any]:
    return {
        "complete": libraries.complete,
        "loaded": list(libraries.loaded),
        "missing": list(libraries.missing),
    }


def diagnostic_report() -> dict[str, Any]:
    """Return a machine-readable report without user-specific filesystem paths."""

    cuda = probe_cuda()
    ct2_visible, ct2_visible_error = _ctranslate2_driver_visible()
    libraries = preload_cuda_libraries()
    ct2_cuda, ct2_error = _ctranslate2_cuda_available()
    devices = []
    for index, name in enumerate(cuda.devices):
        memory = gpu_memory(index)
        devices.append(
            {
                "index": index,
                "name": name,
                "total_memory_mib": cuda.total_memory_mib[index],
                "free_memory_mib": memory.free_mib if memory else None,
                "compute_capability": (
                    cuda.capabilities[index] if index < len(cuda.capabilities) else None
                ),
            }
        )
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
                "driver_visible": ct2_visible,
                "driver_error": ct2_visible_error,
                "error": ct2_error,
                "libraries": _libraries_report(libraries),
            },
            "devices": devices,
            "architectures": list(cuda.architectures),
            "supported_compute_types": _supported_compute_types(),
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
    libraries = ctranslate2.get("libraries")
    if libraries is not None:
        lines.append(
            f"Bibliotecas CUDA do CTranslate2: "
            f"{'completas' if libraries['complete'] else 'incompletas'}"
        )
        if libraries["missing"]:
            lines.append(f"Bibliotecas ausentes: {', '.join(libraries['missing'])}")
    if ctranslate2.get("driver_visible") and not ctranslate2["available"]:
        lines.append(
            "Atenção: o driver expõe a GPU ao CTranslate2, mas as bibliotecas que ele "
            "exige não carregam. A transcrição em GPU falharia."
        )
    if pytorch["error"]:
        lines.append(f"Erro CUDA: {pytorch['error']}")
    if ctranslate2["error"]:
        lines.append(f"Erro CTranslate2/CUDA: {ctranslate2['error']}")
    for device in cuda["devices"]:
        details = [f"{device['total_memory_mib']} MiB"]
        if device.get("free_memory_mib") is not None:
            details.append(f"livre {device['free_memory_mib']} MiB")
        if device.get("compute_capability"):
            details.append(device["compute_capability"])
        lines.append(f"GPU {device['index']}: {device['name']} ({', '.join(details)})")
    architectures = cuda.get("architectures")
    if architectures:
        lines.append(f"Arquiteturas do PyTorch: {', '.join(architectures)}")
    compute_types = cuda.get("supported_compute_types") or {}
    for device_name, label in (("cuda", "GPU"), ("cpu", "CPU")):
        supported = compute_types.get(device_name)
        if supported:
            lines.append(f"Precisões do CTranslate2 ({label}): {', '.join(supported)}")
    lines.append(
        f"Dispositivo automático (openai-whisper): "
        f"{backends['openai-whisper']['automatic_device']}"
    )
    lines.append(
        f"Dispositivo automático (faster-whisper): "
        f"{backends['faster-whisper']['automatic_device']}"
    )
    return lines
