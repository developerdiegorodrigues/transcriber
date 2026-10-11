"""Loader support for the CUDA libraries required by CTranslate2.

CTranslate2 wheels are built against CUDA 12 and resolve ``libcublas.so.12`` and
``libcudnn.so.9`` through the system loader, while PyTorch ships its own CUDA 13
libraries and resolves them internally. On a machine where only the CUDA 13 line is
installed system-wide, the driver probe succeeds but the first inference fails.

Preloading the libraries by absolute path fixes this without ``LD_LIBRARY_PATH``:
once an object is loaded, a later ``dlopen`` of the same soname reuses it instead of
searching the filesystem again.
"""

from __future__ import annotations

import ctypes
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path


CUBLAS_PACKAGE = "nvidia-cublas-cu12"
CUDNN_PACKAGE = "nvidia-cudnn-cu12 (ou a build cu13 que acompanha o PyTorch)"

# Dependencies first: cuDNN's frontend dlopens its own sublibraries by soname, and
# cuBLAS depends on cuBLASLt.
_CUDNN_LIBRARIES = (
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_graph.so.9",
    "libcudnn.so.9",
)
_CUBLAS_LIBRARIES = ("libcublasLt.so.12", "libcublas.so.12")

#: Libraries CTranslate2 cannot run on GPU without, mapped to the package that ships
#: them. Optional sublibraries are preloaded as well but never reported as missing.
REQUIRED_LIBRARIES: dict[str, str] = {
    "libcublas.so.12": CUBLAS_PACKAGE,
    "libcublasLt.so.12": CUBLAS_PACKAGE,
    "libcudnn.so.9": CUDNN_PACKAGE,
}


@dataclass(frozen=True, slots=True)
class CudaLibraryStatus:
    """Outcome of preloading the CUDA libraries CTranslate2 depends on."""

    loaded: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    errors: tuple[str, ...] = field(default=())

    @property
    def complete(self) -> bool:
        return not self.missing

    def describe_missing(self) -> str | None:
        if not self.missing:
            return None
        packages = sorted({REQUIRED_LIBRARIES[name] for name in self.missing})
        return (
            f"bibliotecas CUDA do CTranslate2 ausentes: {', '.join(self.missing)}. "
            f"Instale: {', '.join(packages)}"
        )


def _site_package_roots() -> list[Path]:
    paths = sysconfig.get_paths()
    roots: list[Path] = []
    for key in ("purelib", "platlib"):
        value = paths.get(key)
        if not value:
            continue
        root = Path(value)
        if root.is_dir() and root not in roots:
            roots.append(root)
    return roots


def library_directories() -> list[Path]:
    """Return the ``nvidia/*/lib`` directories bundled inside the environment."""

    directories: list[Path] = []
    for root in _site_package_roots():
        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            continue
        for candidate in sorted(nvidia.glob("*/lib")):
            if candidate.is_dir() and candidate not in directories:
                directories.append(candidate)
    return directories


def _load(path: Path) -> str | None:
    try:
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:  # pragma: no cover - loader/driver dependent
        return str(exc)
    return None


def _already_loaded(name: str) -> bool:
    try:
        ctypes.CDLL(name)
    except OSError:
        return False
    return True


_cached_status: CudaLibraryStatus | None = None


def preload_cuda_libraries(*, refresh: bool = False) -> CudaLibraryStatus:
    """Preload CUDA 12 cuBLAS and cuDNN 9 so CTranslate2 can resolve them.

    The result is cached because loading is a process-wide, idempotent side effect.
    Never raises: callers decide whether a missing library is fatal.
    """

    global _cached_status
    if _cached_status is not None and not refresh:
        return _cached_status

    directories = library_directories()
    loaded: list[str] = []
    errors: list[str] = []
    pending: list[Path] = []
    for name in (*_CUDNN_LIBRARIES, *_CUBLAS_LIBRARIES):
        for directory in directories:
            candidate = directory / name
            if candidate.is_file():
                pending.append(candidate)
                break

    # Two passes: a library whose dependency appears later in the environment still
    # loads on the second attempt.
    remaining = pending
    for _ in range(2):
        failed: list[Path] = []
        for path in remaining:
            error = _load(path)
            if error is None:
                loaded.append(path.name)
            else:
                failed.append(path)
                errors.append(error)
        if not failed:
            break
        remaining = failed

    missing = tuple(
        name
        for name in REQUIRED_LIBRARIES
        if name not in loaded and not _already_loaded(name)
    )
    _cached_status = CudaLibraryStatus(
        loaded=tuple(loaded),
        missing=missing,
        errors=tuple(errors),
    )
    return _cached_status
