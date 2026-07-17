from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import accelerator  # noqa: E402
import distributed as domino_dist  # noqa: E402


class FakeNPU:
    def __init__(self, available=True):
        self.available = available
        self.calls = []

    def is_available(self):
        return self.available

    def set_device(self, value):
        self.calls.append(("set_device", value))

    def synchronize(self, device=None):
        self.calls.append(("synchronize", device))

    def manual_seed_all(self, seed):
        self.calls.append(("manual_seed_all", seed))

    def reset_peak_memory_stats(self, device=None):
        self.calls.append(("reset_peak_memory_stats", device))

    def max_memory_allocated(self, device=None):
        self.calls.append(("max_memory_allocated", device))
        return 1234

    def device_count(self):
        return 8


def test_auto_prefers_cuda_then_npu(monkeypatch):
    fake_npu = FakeNPU(available=True)
    monkeypatch.setattr(accelerator, "_npu_api", lambda required=True: fake_npu)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert accelerator.resolve_backend(requested="auto") == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert accelerator.resolve_backend("auto") == "npu"

    fake_npu.available = False
    assert accelerator.resolve_backend("auto") == "cpu"


def test_auto_falls_back_to_cpu_and_explicit_unavailable_raises(monkeypatch):
    monkeypatch.setattr(accelerator, "_npu_api", lambda required=True: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert accelerator.resolve_backend("auto") == "cpu"
    with pytest.raises(RuntimeError, match="cuda.*not available"):
        accelerator.resolve_backend("cuda")


def test_npu_runtime_helpers(monkeypatch):
    fake_npu = FakeNPU()
    monkeypatch.setattr(accelerator, "_npu_api", lambda required=True: fake_npu)
    monkeypatch.setattr(
        accelerator,
        "get_device",
        lambda local_rank=0, backend="auto": f"npu:{local_rank}",
    )
    monkeypatch.setattr(torch, "device", lambda value: value)

    assert accelerator.set_device(index=3, backend="npu") == "npu:3"
    accelerator.synchronize(3, backend="npu")
    accelerator.manual_seed_all(7, "npu")
    accelerator.reset_peak_memory_stats(3, backend="npu")
    assert accelerator.max_memory_allocated(3, backend="npu") == 1234
    assert accelerator.max_memory_allocated_mb(3, backend="npu") == pytest.approx(
        1234 / (1024**2)
    )
    assert accelerator.device_count("npu") == 8
    assert fake_npu.calls == [
        ("set_device", "npu:3"),
        ("synchronize", 3),
        ("manual_seed_all", 7),
        ("reset_peak_memory_stats", 3),
        ("max_memory_allocated", 3),
        ("max_memory_allocated", 3),
    ]


@pytest.mark.parametrize(
    ("accelerator_name", "expected"),
    [("cpu", "gloo"), ("cuda", "nccl"), ("npu", "hccl")],
)
def test_distributed_backend_mapping(monkeypatch, accelerator_name, expected):
    monkeypatch.setattr(accelerator, "is_backend_available", lambda name: True)
    assert accelerator.distributed_backend(accelerator_name) == expected
    assert accelerator.distributed_backend(expected) == expected


def test_distributed_init_binds_npu_before_hccl_when_explicit(monkeypatch):
    events = []
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "5")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(domino_dist.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(domino_dist, "distributed_backend", lambda backend: "hccl")
    monkeypatch.setattr(
        domino_dist,
        "set_device",
        lambda device, backend: events.append(("set_device", device, backend)),
    )
    monkeypatch.setattr(
        domino_dist.dist,
        "init_process_group",
        lambda **kwargs: events.append(("init_process_group", kwargs)),
    )
    monkeypatch.setattr(
        domino_dist.dist,
        "barrier",
        lambda: events.append(("barrier",)),
    )

    domino_dist.init("npu")

    assert events[0] == ("set_device", 5, "npu")
    assert events[1][0] == "init_process_group"
    assert events[1][1]["backend"] == "hccl"
    assert events[1][1]["init_method"] == "env://"
    assert events[2] == ("barrier",)


def test_distributed_init_defaults_to_nccl_without_npu_binding(monkeypatch):
    events = []
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "5")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(domino_dist.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        domino_dist,
        "set_device",
        lambda device, backend: events.append(("set_device", device, backend)),
    )
    monkeypatch.setattr(
        domino_dist.dist,
        "init_process_group",
        lambda **kwargs: events.append(("init_process_group", kwargs)),
    )
    monkeypatch.setattr(
        domino_dist.dist,
        "barrier",
        lambda: events.append(("barrier",)),
    )

    domino_dist.init()

    assert events[0][0] == "init_process_group"
    assert events[0][1]["backend"] == "nccl"
    assert events[0][1]["init_method"] == "env://"
    assert all(event[0] != "set_device" for event in events)
    assert all(event[0] != "barrier" for event in events)


def test_distributed_init_is_noop_when_already_initialized(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(domino_dist.dist, "is_initialized", lambda: True)
    init = SimpleNamespace(called=False)

    def unexpected_init(**kwargs):
        init.called = True

    monkeypatch.setattr(domino_dist.dist, "init_process_group", unexpected_init)
    domino_dist.init("gloo")
    assert not init.called


def test_distributed_init_single_process_does_not_barrier(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(domino_dist.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(domino_dist, "distributed_backend", lambda backend: "gloo")
    monkeypatch.setattr(domino_dist.dist, "init_process_group", lambda **kwargs: None)
    monkeypatch.setattr(
        domino_dist.dist,
        "barrier",
        lambda: pytest.fail("single-process initialization must not call barrier"),
    )

    domino_dist.init("cpu")
