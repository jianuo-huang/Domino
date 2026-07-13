import os
import warnings
from typing import Any, List, Optional
from torch import distributed as dist
import datetime

from accelerator import distributed_backend, set_device

__all__ = [
    "init",
    "is_initialized",
    "size",
    "rank",
    "local_size",
    "local_rank",
    "is_main",
    "barrier",
    "gather",
    "all_gather",
]


def init(backend: Optional[str] = None) -> None:
    """Initialize a launcher's process group on CUDA, NPU, or CPU.

    ``backend`` accepts accelerator names (``auto``, ``cuda``, ``npu``,
    ``cpu``) as well as native distributed names (``nccl``, ``hccl``,
    ``gloo``).  Existing no-argument CUDA callers remain compatible through
    auto-detection.
    """

    if "RANK" not in os.environ:
        warnings.warn(
            "Environment variable `RANK` is not set. Skipping distributed initialization.",
            stacklevel=2,
        )
        return
    if dist.is_initialized():
        return

    selected_backend = distributed_backend(backend)
    if selected_backend == "nccl":
        set_device(local_rank(), backend="cuda")
    elif selected_backend == "hccl":
        set_device(local_rank(), backend="npu")

    dist.init_process_group(
        backend=selected_backend,
        init_method="env://",
        timeout=datetime.timedelta(hours=2),
    )
    # HCCL establishes device communication lazily on the first collective.
    # Connect while every rank is still together instead of at the final
    # result-merge barrier, where variable-length generations can leave fast
    # and slow ranks more than HCCL's connection timeout apart.
    if size() > 1:
        dist.barrier()


def is_initialized() -> bool:
    return dist.is_initialized()


def size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def rank() -> int:
    return int(os.environ.get("RANK", 0))


def local_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main() -> bool:
    return rank() == 0


def barrier() -> None:
    if not is_initialized():
        return
    dist.barrier()


def gather(obj: Any, dst: int = 0) -> Optional[List[Any]]:
    if not is_initialized():
        return [obj]
    if is_main():
        objs = [None for _ in range(size())]
        dist.gather_object(obj, objs, dst=dst)
        return objs
    else:
        dist.gather_object(obj, dst=dst)
        return None


def all_gather(obj: Any) -> List[Any]:
    if not is_initialized():
        return [obj]
    objs = [None for _ in range(size())]
    dist.all_gather_object(objs, obj)
    return objs
