import time
from types import SimpleNamespace
from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
# Keep the model-specific import used by the upstream CUDA workflow whenever
# available.  Newer Transformers moved this base class to modeling_layers;
# the final nn.Module fallback is only for inference-only older releases.
try:
    from transformers.models.qwen3.modeling_qwen3 import GradientCheckpointingLayer
except ImportError:
    try:
        from transformers.modeling_layers import GradientCheckpointingLayer
    except ImportError:
        GradientCheckpointingLayer = nn.Module
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack

from accelerator import max_memory_allocated_mb, reset_peak_memory_stats, synchronize


def is_domino_projector(projector_type):
    return projector_type == "domino"


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def device_time(device: torch.device | str | int | None = None) -> float:
    """Return a synchronized wall-clock timestamp for accelerator timing."""

    synchronize(device)
    return time.perf_counter()


def cuda_time(device: torch.device | str | int | None = None) -> float:
    """Backward-compatible name for the original CUDA timing helper."""

    return device_time(device)


def manual_gru(
    gru: nn.GRU,
    inputs: torch.Tensor,
    hidden: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a one-layer, unidirectional GRU with basic PyTorch operators.

    CANN 8.0 does not provide a BF16 DynamicGRU kernel.  Expanding the cell
    keeps checkpoint weights and PyTorch's r/z/n gate semantics unchanged while
    allowing torch_npu to dispatch Linear, sigmoid, and tanh independently.
    """

    if gru.num_layers != 1 or gru.bidirectional:
        raise ValueError("manual_gru supports one unidirectional GRU layer only.")
    if not gru.batch_first:
        inputs = inputs.transpose(0, 1)
    batch_size = inputs.shape[0]
    if hidden is None:
        h_state = torch.zeros(
            batch_size,
            gru.hidden_size,
            dtype=inputs.dtype,
            device=inputs.device,
        )
    else:
        if hidden.shape != (1, batch_size, gru.hidden_size):
            raise ValueError(
                "hidden must have shape "
                f"(1, {batch_size}, {gru.hidden_size}), got {tuple(hidden.shape)}."
            )
        h_state = hidden[0]

    outputs = []
    bias_ih = gru.bias_ih_l0 if gru.bias else None
    bias_hh = gru.bias_hh_l0 if gru.bias else None
    for step in range(inputs.shape[1]):
        gi = F.linear(inputs[:, step, :], gru.weight_ih_l0, bias_ih)
        gh = F.linear(h_state, gru.weight_hh_l0, bias_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        reset = torch.sigmoid(i_r + h_r)
        update = torch.sigmoid(i_z + h_z)
        candidate = torch.tanh(i_n + reset * h_n)
        h_state = (1.0 - update) * candidate + update * h_state
        outputs.append(h_state)

    output = torch.stack(outputs, dim=1)
    if not gru.batch_first:
        output = output.transpose(0, 1)
    return output, h_state.unsqueeze(0)


def run_gru(
    gru: nn.GRU,
    inputs: torch.Tensor,
    hidden: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the portable GRU expansion on NPU and native GRU elsewhere."""

    if inputs.device.type == "npu":
        return manual_gru(gru, inputs, hidden)
    if hidden is None:
        return gru(inputs)
    return gru(inputs, hidden)


@torch.inference_mode()
def target_greedy_generate(
    input_ids: torch.Tensor,
    target: nn.Module,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    stop_token_ids: Optional[list[int] | int] = None,
    return_dict: bool = False,
) -> torch.Tensor | SimpleNamespace:
    """Generate directly with the target model using the same cache loop.

    Keeping this baseline independent from the draft model makes memory and
    latency comparisons fair: the baseline process never loads Domino weights.
    """

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            "target_greedy_generate supports input_ids with shape [1, seq_len]."
        )
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")

    device = next(target.parameters()).device
    input_ids = input_ids.to(device)
    if isinstance(stop_token_ids, int):
        stop_token_ids = [stop_token_ids]
    elif stop_token_ids is not None:
        stop_token_ids = list(stop_token_ids)

    reset_peak_memory_stats(device)
    total_start = device_time(device)
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + int(max_new_tokens)
    output_ids = torch.empty(
        (1, max_length), dtype=torch.long, device=device
    )
    output_ids[:, :num_input_tokens] = input_ids
    position_ids = torch.arange(max_length, device=device).unsqueeze(0)
    past_key_values = DynamicCache()

    prefill_start = device_time(device)
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=False,
    )
    output_ids[:, num_input_tokens] = sample(output.logits, temperature)[:, -1]
    time_to_first_token = device_time(device) - prefill_start

    decode_start = device_time(device)
    current = num_input_tokens
    stop_tensor = (
        torch.tensor(stop_token_ids, device=device)
        if stop_token_ids is not None
        else None
    )
    stopped = bool(
        stop_tensor is not None
        and torch.isin(output_ids[:, current], stop_tensor).any().item()
    )
    while current + 1 < max_length and not stopped:
        output = target(
            output_ids[:, current : current + 1],
            position_ids=position_ids[:, current : current + 1],
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=False,
        )
        current += 1
        output_ids[:, current] = sample(output.logits, temperature)[:, -1]
        if stop_tensor is not None:
            stopped = bool(
                torch.isin(output_ids[:, current], stop_tensor).any().item()
            )

    output_ids = output_ids[:, : current + 1]
    total_end = device_time(device)
    if not return_dict:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = total_end - decode_start
    steady_state_tokens = max(num_output_tokens - 1, 0)
    time_per_output_token = (
        total_decode_time / steady_state_tokens
        if steady_state_tokens > 0
        else 0.0
    )
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        target_prefill_time=time_to_first_token,
        draft_setup_time=0.0,
        decode_time=total_decode_time,
        steady_state_decode_time=total_decode_time,
        time_per_output_token=time_per_output_token,
        total_wall_time=total_end - total_start,
        peak_memory_mb=max_memory_allocated_mb(device),
        acceptance_lengths=[],
        steady_state_time_per_output_token=time_per_output_token,
    )


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = self.config.dflash_config.get("target_layer_ids", build_target_layer_ids(config.num_target_layers, config.num_hidden_layers))
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        self.projector_type = self.config.dflash_config.get("projector_type", None)
        self.pure_draft_prefix_len = self.config.dflash_config.get("pure_draft_prefix_len", 0)
        self.emb_dim = self.config.dflash_config["emb_dim"]
        self.gru_hidden_dim = self.config.dflash_config["gru_hidden_dim"]

        if not is_domino_projector(self.projector_type):
            raise ValueError(
                "This reviewer package only supports Domino checkpoints; "
                f"got projector_type={self.projector_type!r}."
            )

        self.prefix_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )

        in_dim = config.hidden_size + self.gru_hidden_dim
        self.embed_proj = nn.Sequential(
            nn.Linear(in_dim, self.emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.emb_dim, config.vocab_size, bias=False),
        )

        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Keep draft inputs in the checkpoint dtype.  This is required for
        # CANN's BF16 kernels and is harmless for the original CUDA workflow,
        # whose target and draft checkpoints use the same dtype in practice.
        draft_dtype = self.fc.weight.dtype
        hidden_states = noise_embedding.to(dtype=draft_dtype)
        target_hidden = target_hidden.to(dtype=draft_dtype)
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        input_ids: torch.Tensor,
        target: nn.Module,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        stop_token_ids: Optional[list[int] | int] = None,
        block_size: Optional[int] = None,
        graph_runner=None,
        use_bias: bool = True,
        return_dict: bool = False,
        track_peak_memory: bool = False,
    ) -> torch.Tensor | SimpleNamespace:
        """Generate with Domino speculative decoding.

        This method currently supports a single sequence on one accelerator, matching the
        draft checkpoints released with this repository.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "spec_generate currently supports input_ids with shape [1, seq_len]."
            )

        target_device = next(target.parameters()).device
        if target_device != self.device:
            raise ValueError(
                "The draft model and target model must be on the same device; "
                f"got draft={self.device}, target={target_device}."
            )

        input_ids = input_ids.to(self.device)
        if track_peak_memory:
            reset_peak_memory_stats(self.device)
        total_start = device_time(self.device)
        block_size = int(block_size or self.block_size)
        mask_token_id = self.mask_token_id
        if mask_token_id is None:
            raise ValueError("The draft model config must define dflash_config.mask_token_id.")

        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]
        elif stop_token_ids is not None:
            stop_token_ids = list(stop_token_ids)

        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + int(max_new_tokens)
        shift_label = bool(
            getattr(self.config, "dflash_config", {}).get("shift_label", False)
        )
        extra_buffer = block_size + 1 if shift_label else block_size

        output_ids = torch.full(
            (1, max_length + extra_buffer),
            mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=self.device).unsqueeze(0)
        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        prefill_start = device_time(self.device)
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=block_size > 1,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        if block_size > 1:
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )
        time_to_first_token = device_time(self.device) - prefill_start

        decode_start = device_time(self.device)
        legacy_decode_start = decode_start
        start = num_input_tokens
        acceptance_lengths: list[int] = []
        draft_prefill = True
        draft_setup_time = 0.0
        prefix_len = int(self.pure_draft_prefix_len)

        # The CUDA path intentionally keeps the upstream loop boundary.  Its
        # final verification block may overrun ``max_length`` into the
        # preallocated scratch area and is trimmed below, which is part of the
        # original CUDA decoding/timing contract.  On NPU, running that final
        # partial block triggers unsupported/out-of-range cache positions, so
        # the portable path stops before ``start == max_length - 1`` and
        # shortens the verification input below.
        is_cuda = self.device.type == "cuda"
        loop_limit = max_length if is_cuda else max_length - 1
        while start < loop_limit:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            k_draft = block_size if shift_label else block_size - 1
            verify_ids = torch.full(
                (1, k_draft + 1),
                mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
            verify_ids[:, 0] = output_ids[:, start]
            verify_position_ids = position_ids[:, start : start + k_draft + 1]

            if block_size > 1:
                draft_setup_start = (
                    device_time(self.device) if draft_prefill else None
                )
                if not is_domino_projector(self.projector_type):
                    raise ValueError(
                        "This package only supports Domino checkpoints; "
                        f"got projector_type={self.projector_type!r}."
                    )
                if not use_bias:
                    raise ValueError("Domino generation requires use_bias=True.")

                draft_dtype = self.prefix_gru.weight_ih_l0.dtype
                noise_embedding = target.model.embed_tokens(block_output_ids).to(
                    dtype=draft_dtype
                )
                parallel_hiddens = self(
                    target_hidden=target_hidden.to(dtype=draft_dtype),
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :, past_key_values_draft.get_seq_length() : start + block_size
                    ],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )
                if not shift_label:
                    parallel_hiddens = parallel_hiddens[:, -block_size + 1 :, :]
                past_key_values_draft.crop(start)

                lm_head_dtype = target.lm_head.weight.dtype
                parallel_hiddens = parallel_hiddens.to(dtype=lm_head_dtype)
                base_logits = target.lm_head(parallel_hiddens)
                if prefix_len > 0:
                    prefix_token_ids = sample(base_logits[:, :prefix_len], temperature)
                    verify_ids[:, 1 : 1 + prefix_len] = prefix_token_ids

                if graph_runner is not None:
                    graph_prefix_ids = verify_ids[:, : 1 + prefix_len].contiguous()
                    graph_parallel_hiddens = parallel_hiddens[
                        :, prefix_len:k_draft, :
                    ].contiguous()
                    graph_base_logits = base_logits[:, prefix_len:k_draft, :].contiguous()
                    verify_ids[:, 1 + prefix_len :] = graph_runner(
                        graph_prefix_ids,
                        graph_parallel_hiddens,
                        graph_base_logits,
                    )
                else:
                    realized_prefix_ids = verify_ids[:, : 1 + prefix_len]
                    realized_prefix_embeds = target.model.embed_tokens(
                        realized_prefix_ids
                    ).to(dtype=self.prefix_gru.weight_ih_l0.dtype)
                    _, gru_hidden = run_gru(self.prefix_gru, realized_prefix_embeds)

                    for i in range(prefix_len, k_draft):
                        z_i = parallel_hiddens[:, i : i + 1, :]
                        s_i = gru_hidden.transpose(0, 1)

                        bias = self.embed_proj(torch.cat([z_i, s_i], dim=-1))
                        current_token_id = sample(
                            base_logits[:, i : i + 1, :]
                            + bias.to(dtype=base_logits.dtype),
                            temperature,
                        )
                        verify_ids[:, i + 1 : i + 2] = current_token_id

                        if i + 1 < k_draft:
                            new_embed = target.model.embed_tokens(current_token_id)
                            new_embed = new_embed.to(
                                dtype=self.prefix_gru.weight_ih_l0.dtype
                            )
                            _, gru_hidden = run_gru(
                                self.prefix_gru, new_embed, gru_hidden
                            )

                if draft_prefill:
                    draft_prefill = False
                    draft_setup_end = device_time(self.device)
                    draft_setup_time = draft_setup_end - draft_setup_start
                    # Reuse the same synchronized timestamp so the CUDA
                    # timing path does not add an extra synchronization that
                    # the upstream implementation never performed.
                    legacy_decode_start = draft_setup_end

            # Each verified input can produce one new output token.  Do not run
            # target positions whose predictions would be discarded by the
            # max-new-tokens budget, especially in the final partial block.
            verify_length = (
                k_draft + 1
                if is_cuda
                else min(k_draft + 1, max_length - start - 1)
            )
            round_verify_ids = verify_ids[:, :verify_length]
            round_position_ids = verify_position_ids[:, :verify_length]
            output = target(
                round_verify_ids,
                position_ids=round_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=block_size > 1,
            )

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (round_verify_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            correction = posterior[:, acceptance_length : acceptance_length + 1]

            output_ids[:, start : start + acceptance_length + 1] = round_verify_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = correction[:, 0]

            acceptance_lengths.append(int(acceptance_length) + 1)
            start += int(acceptance_length) + 1
            past_key_values_target.crop(start)
            if block_size > 1:
                target_hidden = extract_context_feature(
                    output.hidden_states, self.target_layer_ids
                )[:, : acceptance_length + 1, :]

            if stop_token_ids is not None:
                stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
                if not is_cuda:
                    # Include the correction token produced at ``start`` so
                    # the NPU path can stop without entering another partial
                    # verification round.
                    stop_check_end = min(start + 1, max_length)
                    should_stop = torch.isin(
                        output_ids[:, num_input_tokens:stop_check_end], stop_tensor
                    ).any()
                else:
                    # Keep the original CUDA boundary: ``start`` is the
                    # correction position and is checked by the final trim
                    # below, not by this in-loop early-exit check.
                    should_stop = torch.isin(
                        output_ids[:, num_input_tokens:start], stop_tensor
                    ).any()
                if should_stop:
                    break

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != mask_token_id]
        if stop_token_ids is not None:
            stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_tensor
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0].item() + 1
                ]

        if not return_dict:
            return output_ids

        total_end = device_time(self.device)
        num_output_tokens = output_ids.shape[1] - num_input_tokens
        total_decode_time = total_end - decode_start
        legacy_decode_time = total_end - legacy_decode_start
        steady_state_decode_time = max(total_decode_time - draft_setup_time, 0.0)
        steady_state_tokens = max(num_output_tokens - 1, 0)
        steady_state_time_per_output_token = (
            steady_state_decode_time / steady_state_tokens
            if steady_state_tokens > 0
            else 0.0
        )
        time_per_output_token = (
            legacy_decode_time / num_output_tokens
            if num_output_tokens > 0
            else 0.0
        )
        return SimpleNamespace(
            output_ids=output_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            time_to_first_token=time_to_first_token,
            target_prefill_time=time_to_first_token,
            draft_setup_time=draft_setup_time,
            decode_time=total_decode_time,
            steady_state_decode_time=steady_state_decode_time,
            time_per_output_token=time_per_output_token,
            steady_state_time_per_output_token=steady_state_time_per_output_token,
            total_wall_time=total_end - total_start,
            peak_memory_mb=(
                max_memory_allocated_mb(self.device) if track_peak_memory else 0.0
            ),
            acceptance_lengths=acceptance_lengths,
        )
