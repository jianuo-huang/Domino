import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import dflash  # noqa: E402


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("batch_first", [False, True])
def test_manual_gru_matches_native_gru_cpu_fp32(bias, batch_first):
    torch.manual_seed(7)
    gru = nn.GRU(
        input_size=5,
        hidden_size=7,
        num_layers=1,
        bias=bias,
        batch_first=batch_first,
    ).eval()
    inputs = torch.randn((3, 4, 5) if batch_first else (4, 3, 5))
    hidden = torch.randn(1, 3, 7)

    expected_output, expected_hidden = gru(inputs, hidden)
    actual_output, actual_hidden = dflash.manual_gru(gru, inputs, hidden)

    torch.testing.assert_close(actual_output, expected_output, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=1e-6, atol=1e-6)


class DummyCausalModel(nn.Module):
    """Minimal deterministic causal LM: every token predicts token + 1."""

    def __init__(self, vocab_size=10):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self.calls = []

    def forward(self, input_ids, **kwargs):
        self.calls.append((input_ids.detach().clone(), kwargs))
        logits = torch.full(
            (input_ids.shape[0], 1, self.vocab_size),
            -1_000.0,
            dtype=self.anchor.dtype,
            device=input_ids.device,
        )
        next_ids = (input_ids[:, -1] + 1) % self.vocab_size
        logits.scatter_(2, next_ids[:, None, None], 1_000.0)
        return SimpleNamespace(logits=logits)


class DummyDraftModel(nn.Module):
    """Minimal draft wrapper for exercising block-size-one generation."""

    spec_generate = dflash.DFlashDraftModel.spec_generate

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.block_size = 1
        self.mask_token_id = 99
        self.config = SimpleNamespace(dflash_config={"shift_label": False})
        self.pure_draft_prefix_len = 0

    @property
    def device(self):
        return self.anchor.device


class BatchDivergentTarget(nn.Module):
    """Target whose low-margin block argmax differs from single-token decode."""

    def __init__(self, vocab_size=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.lm_head = nn.Linear(vocab_size, vocab_size, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(vocab_size))
            self.lm_head.weight.copy_(torch.eye(vocab_size))
        self.model = SimpleNamespace(embed_tokens=self.embedding)
        self.last_cache = None

    def forward(self, input_ids, **kwargs):
        cache = kwargs["past_key_values"]
        cache_values = torch.zeros(
            1, 1, input_ids.shape[1], 1, device=input_ids.device
        )
        cache.update(cache_values, cache_values, layer_idx=0)
        self.last_cache = cache

        logits_input = (
            input_ids[:, -1:]
            if kwargs.get("logits_to_keep") == 1
            else input_ids
        )
        next_ids = (logits_input + 1) % (self.vocab_size - 2)
        logits = torch.full(
            (*next_ids.shape, self.vocab_size),
            -10.0,
            device=input_ids.device,
        )
        logits.scatter_(2, next_ids.unsqueeze(-1), 10.0)
        if input_ids.shape[1] > 1 and kwargs.get("logits_to_keep") is None:
            # Simulate an NPU BF16 batch kernel changing argmax at low margin.
            alternate = (next_ids + 1) % (self.vocab_size - 2)
            logits.fill_(-10.0)
            logits.scatter_(2, next_ids.unsqueeze(-1), 9.9)
            logits.scatter_(2, alternate.unsqueeze(-1), 10.0)

        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=logits, hidden_states=(hidden, hidden))


class BatchDivergenceDraft(nn.Module):
    """Drafts the target's ordinary sequential token-plus-one continuation."""

    spec_generate = dflash.DFlashDraftModel.spec_generate

    def __init__(self, vocab_size=12):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.block_size = 3
        self.mask_token_id = vocab_size - 1
        self.config = SimpleNamespace(dflash_config={"shift_label": False})
        self.pure_draft_prefix_len = 2
        self.projector_type = "domino"
        self.target_layer_ids = [0]
        self.prefix_gru = nn.GRU(vocab_size, vocab_size, batch_first=True)
        self.embed_proj = nn.Linear(2 * vocab_size, vocab_size, bias=False)
        self.seen_noise_dtype = None
        self.seen_target_hidden_dtype = None

    @property
    def device(self):
        return self.anchor.device

    def forward(self, noise_embedding, **kwargs):
        self.seen_noise_dtype = noise_embedding.dtype
        self.seen_target_hidden_dtype = kwargs["target_hidden"].dtype
        current = noise_embedding[:, 0].argmax(dim=-1)
        offsets = torch.arange(
            self.block_size, device=noise_embedding.device
        ).unsqueeze(0)
        tokens = (current.unsqueeze(1) + offsets) % (noise_embedding.shape[-1] - 2)
        return torch.nn.functional.one_hot(
            tokens, num_classes=noise_embedding.shape[-1]
        ).to(noise_embedding.dtype)


def test_target_greedy_generate_respects_output_shape_and_limit():
    target = DummyCausalModel()

    output_ids = dflash.target_greedy_generate(
        input_ids=torch.tensor([[5, 6]]),
        target=target,
        max_new_tokens=4,
    )

    assert output_ids.shape == (1, 6)
    assert output_ids.tolist() == [[5, 6, 7, 8, 9, 0]]
    assert len(target.calls) == 4
    assert target.calls[0][0].shape == (1, 2)
    assert all(call[0].shape == (1, 1) for call in target.calls[1:])


def test_target_greedy_generate_stops_at_eos_and_returns_metrics(monkeypatch):
    target = DummyCausalModel()
    timestamps = iter([10.0, 11.0, 12.0, 13.0, 14.0])
    monkeypatch.setattr(dflash, "device_time", lambda device=None: next(timestamps))
    monkeypatch.setattr(dflash, "max_memory_allocated_mb", lambda device=None: 321.5)

    result = dflash.target_greedy_generate(
        input_ids=torch.tensor([[0, 1]]),
        target=target,
        max_new_tokens=5,
        stop_token_ids=3,
        return_dict=True,
    )

    assert result.output_ids.tolist() == [[0, 1, 2, 3]]
    assert result.num_input_tokens == 2
    assert result.num_output_tokens == 2
    assert result.time_to_first_token == pytest.approx(1.0)
    assert result.target_prefill_time == pytest.approx(1.0)
    assert result.draft_setup_time == pytest.approx(0.0)
    assert result.decode_time == pytest.approx(1.0)
    assert result.steady_state_decode_time == pytest.approx(1.0)
    assert result.time_per_output_token == pytest.approx(1.0)
    assert result.total_wall_time == pytest.approx(4.0)
    assert result.peak_memory_mb == pytest.approx(321.5)
    assert result.acceptance_lengths == []
    assert not hasattr(result, "sequential_fallbacks")
    assert not hasattr(result, "sequential_catchup_mismatches")


def test_spec_generate_stops_when_current_posterior_is_eos():
    target = DummyCausalModel()
    draft = DummyDraftModel()

    output_ids = draft.spec_generate(
        input_ids=torch.tensor([[0, 1]]),
        target=target,
        max_new_tokens=5,
        stop_token_ids=3,
    )

    assert output_ids.tolist() == [[0, 1, 2, 3]]
    assert len(target.calls) == 2
    assert target.calls[-1][0].tolist() == [[2]]


def test_spec_generate_does_not_verify_beyond_max_length():
    target = DummyCausalModel()
    draft = DummyDraftModel()

    output_ids = draft.spec_generate(
        input_ids=torch.tensor([[0, 1]]),
        target=target,
        max_new_tokens=1,
    )

    assert output_ids.tolist() == [[0, 1, 2]]
    assert len(target.calls) == 1


def test_spec_generate_uses_block_verification_by_default():
    input_ids = torch.tensor([[0, 1]])
    expected = dflash.target_greedy_generate(
        input_ids=input_ids,
        target=BatchDivergentTarget(),
        max_new_tokens=5,
    )

    target = BatchDivergentTarget()
    result = BatchDivergenceDraft().spec_generate(
        input_ids=input_ids,
        target=target,
        max_new_tokens=5,
        return_dict=True,
    )
    assert not torch.equal(result.output_ids, expected)
    assert target.last_cache.get_seq_length() == result.output_ids.shape[1] - 1
    assert not hasattr(result, "sequential_fallbacks")
    assert not hasattr(result, "sequential_catchup_mismatches")
    assert "sequential_fallback_margin" not in inspect.signature(
        dflash.DFlashDraftModel.spec_generate
    ).parameters

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        BatchDivergenceDraft().spec_generate(
            input_ids=input_ids,
            target=BatchDivergentTarget(),
            max_new_tokens=2,
            sequential_fallback_margin=-1,
        )


def test_fp32_target_with_bf16_draft_casts_boundaries():
    target = BatchDivergentTarget().float()
    draft = BatchDivergenceDraft().to(dtype=torch.bfloat16)

    result = draft.spec_generate(
        input_ids=torch.tensor([[0, 1]]),
        target=target,
        max_new_tokens=5,
        return_dict=True,
    )

    assert result.output_ids.dtype == torch.long
    assert draft.seen_noise_dtype == torch.bfloat16
    assert draft.seen_target_hidden_dtype == torch.bfloat16
    assert target.lm_head.weight.dtype == torch.float32


def test_portable_benchmark_import_and_help_do_not_import_triton():
    child_code = f"""
import builtins
import importlib
import runpy
import sys

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'kernel.domino':
        raise AssertionError(f'unexpected CUDA-only kernel import: {{name}}')
    if name == 'triton' or name.startswith('triton.'):
        raise ModuleNotFoundError(f'No module named {{name!r}}', name=name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
benchmark = importlib.import_module('benchmark_portable')
assert benchmark.TARGET_DTYPES['bfloat16'] == __import__('torch').bfloat16
assert benchmark.TARGET_DTYPES['float32'] == __import__('torch').float32
choice = benchmark._new_choice(0, 'domino', 16)
assert 'sequential_fallbacks' not in choice
assert 'sequential_catchup_mismatches' not in choice
assert 'kernel.domino' not in sys.modules
sys.modules.pop('benchmark_portable', None)
sys.argv = ['benchmark_portable.py', '--help']
try:
    runpy.run_path({str(CODE_DIR / 'benchmark_portable.py')!r}, run_name='__main__')
except SystemExit as exc:
    if exc.code != 0:
        raise
assert 'kernel.domino' not in sys.modules
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(CODE_DIR), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--device-backend" in completed.stdout
    assert "--sequential-fallback-margin" not in completed.stdout
    assert "--target-dtype" in completed.stdout
    assert "float32" in completed.stdout


def test_portable_benchmark_rejects_removed_sequential_fallback_option():
    completed = subprocess.run(
        [
            sys.executable,
            str(CODE_DIR / "benchmark_portable.py"),
            "--sequential-fallback-margin",
            "-1",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(CODE_DIR), os.environ.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 2
    assert "unrecognized arguments: --sequential-fallback-margin -1" in completed.stderr
