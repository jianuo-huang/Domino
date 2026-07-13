"""Small CUDA/NPU device abstraction used by the Hugging Face runner.

The module deliberately imports ``torch_npu`` lazily.  This keeps the
original CUDA path usable in environments where the Ascend extension is not
installed, while still registering ``torch.npu`` before an NPU device is
created.
"""

from __future__ import annotations

import importlib
import random
from types import ModuleType
from typing import Any, Optional, Union

import torch


DeviceLike = Union[int, str, torch.device]
ACCELERATOR_BACKENDS = ("auto", "cpu", "cuda", "npu")
_DISTRIBUTED_BACKENDS = {"cpu": "gloo", "cuda": "nccl", "npu": "hccl"}

__all__ = [
    "ACCELERATOR_BACKENDS",
    "configure_device",
    "device",
    "device_count",
    "distributed_backend",
    "get_backend",
    "get_device",
    "get_distributed_backend",
    "import_torch_npu",
    "is_backend_available",
    "manual_seed_all",
    "max_memory_allocated",
    "max_memory_allocated_mb",
    "reset_peak_memory_stats",
    "resolve_backend",
    "seed_all",
    "set_device",
    "synchronize",
]


def import_torch_npu(required: bool = True) -> Optional[ModuleType]:
    """Import and return ``torch_npu`` without breaking non-Ascend hosts."""

    try:
        return importlib.import_module("torch_npu")
    except (ImportError, OSError, RuntimeError) as exc:
        if required:
            raise RuntimeError(
                "The NPU backend was requested, but torch_npu could not be "
                "imported. Activate the Ascend environment and source CANN's "
                "set_env.sh before starting the process."
            ) from exc
        return None


def _npu_api(required: bool = True) -> Optional[Any]:
    api = getattr(torch, "npu", None)
    if api is None:
        import_torch_npu(required=required)
        api = getattr(torch, "npu", None)
    if api is None and required:
        raise RuntimeError("torch_npu was imported but torch.npu is unavailable")
    return api


def _normalize_backend(backend: Optional[str]) -> str:
    normalized = "auto" if backend is None else str(backend).lower()
    if normalized not in ACCELERATOR_BACKENDS:
        choices = ", ".join(ACCELERATOR_BACKENDS)
        raise ValueError(f"Unsupported accelerator backend {backend!r}; choose one of: {choices}")
    return normalized


def is_backend_available(backend: str) -> bool:
    """Return whether an accelerator backend is usable in this process."""

    backend = _normalize_backend(backend)
    if backend == "auto":
        return True
    if backend == "cpu":
        return True
    if backend == "cuda":
        return bool(torch.cuda.is_available())

    api = _npu_api(required=False)
    return bool(api is not None and api.is_available())


def resolve_backend(requested: Optional[str] = "auto") -> str:
    """Resolve ``auto`` and validate an explicitly requested backend.

    NPU is checked before CUDA so an Ascend host with compatibility CUDA
    packages installed still selects its native device.  CPU is a deliberate
    final fallback for unit tests and diagnostic commands.
    """

    backend = _normalize_backend(requested)
    if backend == "auto":
        for candidate in ("npu", "cuda"):
            if is_backend_available(candidate):
                return candidate
        return "cpu"

    if not is_backend_available(backend):
        raise RuntimeError(f"The requested {backend!r} backend is not available")
    return backend


# A short alias is useful at CLI setup call sites.
get_backend = resolve_backend


def device(backend: Optional[str] = "auto", index: int = 0) -> torch.device:
    """Build a validated torch device for ``backend`` and local device index."""

    resolved = resolve_backend(backend)
    if resolved == "cpu":
        return torch.device("cpu")
    return torch.device(f"{resolved}:{index}")


def get_device(local_rank: int = 0, backend: Optional[str] = "auto") -> torch.device:
    """Return the validated device assigned to a distributed local rank."""

    return device(backend=backend, index=local_rank)


def _backend_for_device(value: Optional[DeviceLike], backend: Optional[str]) -> str:
    if backend is not None:
        return resolve_backend(backend)
    if isinstance(value, torch.device):
        return resolve_backend(value.type)
    if isinstance(value, str):
        device_type = value.split(":", maxsplit=1)[0].lower()
        if device_type in ACCELERATOR_BACKENDS:
            return resolve_backend(device_type)
    return resolve_backend("auto")


def set_device(
    device: DeviceLike = 0,
    backend: Optional[str] = None,
    *,
    index: Optional[int] = None,
) -> torch.device:
    """Select the process-local accelerator and return its torch device."""

    # ``index`` keeps compatibility with explicit keyword-based launch code;
    # normal callers pass the result of get_device directly.
    if index is not None:
        device = index
    resolved = _backend_for_device(device, backend)
    selected = (
        get_device(local_rank=device, backend=resolved)
        if isinstance(device, int)
        else torch.device(device)
    )
    if resolved == "cuda":
        torch.cuda.set_device(selected)
    elif resolved == "npu":
        _npu_api().set_device(selected)
    return selected


def configure_device(local_rank: int = 0, backend: Optional[str] = "auto") -> torch.device:
    """Alias with names matching distributed launch terminology."""

    return set_device(get_device(local_rank=local_rank, backend=backend))


def device_count(backend: Optional[str] = "auto") -> int:
    resolved = resolve_backend(backend)
    if resolved == "cuda":
        return int(torch.cuda.device_count())
    if resolved == "npu":
        return int(_npu_api().device_count())
    return 1


def synchronize(
    device: Optional[DeviceLike] = None, backend: Optional[str] = None
) -> None:
    """Wait for queued work on one device; CPU is intentionally a no-op."""

    resolved = _backend_for_device(device, backend)
    if resolved == "cuda":
        torch.cuda.synchronize(device=device)
    elif resolved == "npu":
        _npu_api().synchronize(device=device)


def seed_all(seed: int, backend: Optional[str] = "auto") -> None:
    """Seed Python, PyTorch CPU, and the selected accelerator RNGs."""

    random.seed(seed)
    torch.manual_seed(seed)
    resolved = resolve_backend(backend)
    if resolved == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif resolved == "npu":
        _npu_api().manual_seed_all(seed)


def manual_seed_all(seed: int, backend: Optional[str] = "auto") -> None:
    """Public spelling matching torch accelerator RNG APIs."""

    seed_all(seed=seed, backend=backend)


def reset_peak_memory_stats(
    device: Optional[DeviceLike] = None, backend: Optional[str] = None
) -> None:
    """Reset peak allocated-memory accounting for the selected accelerator."""

    resolved = _backend_for_device(device, backend)
    if resolved == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)
    elif resolved == "npu":
        _npu_api().reset_peak_memory_stats(device=device)


def max_memory_allocated(
    device: Optional[DeviceLike] = None, backend: Optional[str] = None
) -> int:
    """Return peak allocated bytes, or zero for the CPU backend."""

    resolved = _backend_for_device(device, backend)
    if resolved == "cuda":
        return int(torch.cuda.max_memory_allocated(device=device))
    if resolved == "npu":
        return int(_npu_api().max_memory_allocated(device=device))
    return 0


def max_memory_allocated_mb(
    device: Optional[DeviceLike] = None, backend: Optional[str] = None
) -> float:
    """Return peak allocated memory in binary megabytes (MiB)."""

    return max_memory_allocated(device=device, backend=backend) / (1024**2)


def distributed_backend(backend: Optional[str] = "auto") -> str:
    """Map an accelerator backend to its torch.distributed backend.

    Direct distributed backend names are also accepted so callers can
    override auto-detection without losing the simple ``init(backend=...)``
    interface.
    """

    normalized = "auto" if backend is None else str(backend).lower()
    if normalized in _DISTRIBUTED_BACKENDS.values():
        return normalized
    return _DISTRIBUTED_BACKENDS[resolve_backend(normalized)]


get_distributed_backend = distributed_backend
