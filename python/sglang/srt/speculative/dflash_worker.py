import logging
import math
import os
import time
from copy import deepcopy
from typing import Optional, Union

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import get_last_loc
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import (
    can_dflash_use_fused_qkv_proj,
    is_dflash_sampling_verify_available,
    parse_dflash_draft_config,
    resolve_dflash_verify_mask_policy,
)
from sglang.srt.speculative.dflash_v5_kernels import (
    compute_silu_sum,
    fused_gru_cell,
    fused_gru_cell_from_table,
    fused_mid_fc2_argmax,
    fused_silu_fc2_argmax,
    fused_silu_fc2_argmax_with_value,
    fused_silu_fc2_candidate_argmax,
    fused_silu_fc2_candidate_argmax_with_value,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_FusedKVMaterializeHelper = None


def _get_fused_kv_materialize_helper():
    global _FusedKVMaterializeHelper
    if _FusedKVMaterializeHelper is None:
        from sglang.srt.speculative.triton_ops.fused_kv_materialize import (
            FusedKVMaterializeHelper,
        )

        _FusedKVMaterializeHelper = FusedKVMaterializeHelper
    return _FusedKVMaterializeHelper


class DFlashWorker:
    """DFlash speculative decoding worker (spec-v1, tp>=1/pp=1)."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.draft_window_size: Optional[int] = (
            int(server_args.speculative_dflash_draft_window_size)
            if server_args.speculative_dflash_draft_window_size is not None
            else None
        )
        self.use_compact_draft_cache = self.draft_window_size is not None
        self.device = target_worker.device

        self._warned_sampling_fallback = False
        self._logged_first_verify = False

        # Draft runner (separate KV cache + attention backend).
        # Without draft windowing, the draft worker aliases the target request->token
        # mapping and allocation state. With draft windowing enabled, the draft worker
        # keeps a private compact req->token table over the same global KV index space,
        # so radix-cache/prefix-hit KV remains reusable while draft attention sees only
        # the recent window.
        target_req_to_token_pool, target_token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        shared_req_to_token_pool = (
            None if self.use_compact_draft_cache else target_req_to_token_pool
        )
        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        draft_backend = draft_server_args.speculative_draft_attention_backend
        supported_draft_backends = ("flashinfer", "fa3", "fa4", "triton")
        if draft_backend is None:
            draft_backend, _ = draft_server_args.get_attention_backends()
        if draft_backend is None:
            # Use triton on ROCm (no FlashInfer), flashinfer on CUDA
            import torch as _torch

            draft_backend = "triton" if _torch.version.hip else "flashinfer"
        elif draft_backend == "trtllm_mha":
            import torch as _torch

            _fb = "triton" if _torch.version.hip else "flashinfer"
            logger.warning(
                "DFLASH draft worker does not support 'trtllm_mha' because the "
                "draft path requires non-causal attention. Falling back to "
                "'%s'.",
                _fb,
            )
            draft_backend = _fb
        elif draft_backend not in supported_draft_backends:
            import torch as _torch

            _fb = "triton" if _torch.version.hip else "flashinfer"
            logger.warning(
                "DFLASH draft worker only supports attention_backend in %s for now, "
                "but got %r. Falling back to '%s'.",
                supported_draft_backends,
                draft_backend,
                _fb,
            )
            draft_backend = _fb
        # Make the draft worker backend explicit and self-contained (no further overrides).
        draft_server_args.speculative_draft_attention_backend = None
        draft_server_args.prefill_attention_backend = None
        draft_server_args.decode_attention_backend = None
        draft_server_args.attention_backend = draft_backend
        # Keep draft context length aligned with the target.
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )
        saved_server_args = get_global_server_args()
        self.draft_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=0,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=shared_req_to_token_pool,
            token_to_kv_pool_allocator=target_token_to_kv_pool_allocator,
            memory_pool_config=target_worker.model_runner.memory_pool_config,
        )
        set_global_server_args_for_scheduler(saved_server_args)
        self.draft_model_runner = self.draft_worker.model_runner
        self.draft_model = self.draft_model_runner.model
        draft_config = parse_dflash_draft_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config
        )
        if server_args.speculative_num_draft_tokens is None:
            # Should not happen (ServerArgs should have inferred it), but keep a fallback.
            self.block_size = int(draft_config.resolve_block_size(default=16))
        else:
            self.block_size = int(server_args.speculative_num_draft_tokens)
            model_block_size = draft_config.block_size
            if model_block_size is None:
                model_block_size = getattr(self.draft_model, "block_size", None)
            if model_block_size is not None and int(model_block_size) != int(
                self.block_size
            ):
                logger.warning(
                    "DFLASH block size mismatch: using speculative_num_draft_tokens=%s but draft config block_size=%s.",
                    self.block_size,
                    model_block_size,
                )

        self._mask_token = draft_config.mask_token
        self._mask_token_id_override = draft_config.mask_token_id
        self._mask_token_id = self._resolve_mask_token_id(
            mask_token=self._mask_token,
            mask_token_id=self._mask_token_id_override,
        )
        if self.tp_rank == 0:
            logger.info(
                "Initialized DFLASH draft runner. attention_backend=%s, model=%s, block_size=%s, draft_window_size=%s, compact_cache=%s",
                getattr(draft_server_args, "attention_backend", None),
                self.draft_model.__class__.__name__,
                self.block_size,
                self.draft_window_size,
                self.use_compact_draft_cache,
            )
            logger.info(
                "DFLASH draft runner ready. mask_token=%s, mask_token_id=%s, mask_token_id_override=%s",
                self._mask_token,
                self._mask_token_id,
                self._mask_token_id_override,
            )

        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        self._draft_block_ids_buf: Optional[torch.Tensor] = None  # [cap_bs, block_size]
        self._draft_block_positions_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_tokens_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_end_buf: Optional[torch.Tensor] = None  # [cap_bs]
        self._draft_seq_lens_cpu_buf: Optional[torch.Tensor] = None  # [cap_bs] on CPU
        self._draft_block_spec_info = DFlashVerifyInput(
            draft_token=torch.empty((0,), dtype=torch.long, device=self.device),
            positions=torch.empty((0,), dtype=torch.int64, device=self.device),
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._draft_greedy_gathered_max_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gathered_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_local_pair_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gathered_pair_buf: Optional[torch.Tensor] = None
        self._draft_greedy_selected_ids_f32_buf: Optional[torch.Tensor] = None
        self._draft_greedy_pair_cap: int = 0
        self._draft_greedy_pair_tp_size: int = 0
        self._draft_greedy_gather_cap: int = 0
        self._draft_greedy_best_rank_buf: Optional[torch.Tensor] = None
        self._draft_greedy_rank_index_buf: Optional[torch.Tensor] = None
        self._draft_greedy_selected_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_index_cap: int = 0

        self._use_fused_kv_materialize = is_cuda()
        self._fused_kv_helper: Optional[object] = None
        if self._use_fused_kv_materialize:
            self._init_fused_kv_helper()

    def _init_fused_kv_helper(self) -> None:
        """Initialize the fused KV materialization helper with pre-stacked weights."""
        try:
            layers = self.draft_model.layers
            fused_disable_reason: Optional[str] = None

            if len(layers) == 0:
                fused_disable_reason = "no layers found"

            for layer_idx, layer in enumerate(layers):
                attn = layer.self_attn
                eligible, reason = can_dflash_use_fused_qkv_proj(attn.qkv_proj)
                if not eligible:
                    fused_disable_reason = f"{reason}: layer={layer_idx}"
                    break

                # Keep semantics aligned with set_kv_buffer scaling behavior.
                k_scale = getattr(attn.attn, "k_scale", None)
                v_scale = getattr(attn.attn, "v_scale", None)
                if k_scale is not None and not math.isclose(float(k_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit k_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, k_scale={k_scale}"
                    )
                    break
                if v_scale is not None and not math.isclose(float(v_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit v_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, v_scale={v_scale}"
                    )
                    break

                rope_is_neox_style = bool(
                    getattr(attn.rotary_emb, "is_neox_style", True)
                )
                if not rope_is_neox_style:
                    fused_disable_reason = (
                        "non-neox RoPE is not supported for fused KV path: "
                        f"layer={layer_idx}, rope_is_neox_style={rope_is_neox_style}"
                    )
                    break

            if fused_disable_reason is not None:
                if self.tp_rank == 0:
                    logger.info(
                        "DFLASH fused KV materialization disabled: %s",
                        fused_disable_reason,
                    )
                self._use_fused_kv_materialize = False
                self._fused_kv_helper = None
                return

            FusedKVMaterializeHelper = _get_fused_kv_materialize_helper()
            first_attn = layers[0].self_attn
            rotary_emb = first_attn.rotary_emb

            self._fused_kv_helper = FusedKVMaterializeHelper(
                layers=layers,
                rotary_emb=rotary_emb,
                num_kv_heads=first_attn.num_kv_heads,
                head_dim=first_attn.head_dim,
                device=self.device,
            )
            if self.tp_rank == 0:
                logger.info(
                    "DFLASH fused KV materialization enabled. "
                    "n_layers=%d, num_kv_heads=%d, head_dim=%d",
                    len(layers),
                    first_attn.num_kv_heads,
                    first_attn.head_dim,
                )
        except Exception as e:
            logger.warning(
                "DFLASH fused KV initialization failed, falling back to sequential path: %s",
                e,
            )
            self._use_fused_kv_materialize = False
            self._fused_kv_helper = None

    def _ensure_draft_block_buffers(self, bs: int) -> None:
        cap = (
            0
            if self._draft_block_ids_buf is None
            else int(self._draft_block_ids_buf.shape[0])
        )
        if cap >= int(bs):
            return

        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))
        device = self.device
        block_size = int(self.block_size)
        self._draft_block_ids_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_positions_buf = torch.empty(
            (new_cap, block_size), dtype=torch.int64, device=device
        )
        self._draft_block_tokens_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_end_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device=device
        )
        self._draft_seq_lens_cpu_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device="cpu"
        )

    def __getattr__(self, name):
        # Delegate anything not implemented yet to the target worker.
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        # The target worker owns the shared KV allocator/cache. For the compact
        # sliding-window path, the draft req->token view is rebuilt from committed
        # target state before each draft forward, so there is nothing persistent
        # to flush here.
        pass

    def _gather_req_to_token_masked(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        pos2d: torch.Tensor,
        mask: torch.Tensor,
        context: str,
    ) -> torch.Tensor:
        if pos2d.ndim != 2:
            raise RuntimeError(
                f"{context} expected 2D positions, got shape={tuple(pos2d.shape)}."
            )
        if mask.shape != pos2d.shape:
            raise RuntimeError(
                f"{context} mask/position shape mismatch: {tuple(mask.shape)} vs {tuple(pos2d.shape)}."
            )

        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        table_width = int(req_to_token.shape[1])
        if table_width <= 0:
            if bool(mask.any().item()):
                raise RuntimeError(
                    f"{context} req_to_token table is empty but gather mask is non-empty."
                )
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        # Only the masked-off rectangular padding can be out of range in the normal
        # ragged-batch case. Replace those don't-care columns with a valid in-range
        # position before the gather so the kernel only sees real positions.
        safe_pos2d = pos2d.masked_fill(~mask, 0)
        return req_to_token[req_pool_indices[:, None], safe_pos2d][mask].to(torch.int64)

    def _gather_req_to_token_segments(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        start: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        lengths = lengths.to(torch.int64)
        if lengths.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        max_len = int(lengths.max().item())
        if max_len <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)
        offsets = torch.arange(
            max_len, device=self.device, dtype=torch.int64
        ).unsqueeze(0)
        if start is None:
            pos2d = offsets.expand(req_pool_indices.shape[0], -1)
        else:
            pos2d = start.to(torch.int64).unsqueeze(1) + offsets
        mask = offsets < lengths.unsqueeze(1)
        return self._gather_req_to_token_masked(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            pos2d=pos2d,
            mask=mask,
            context="DFLASH req_to_token segment gather",
        )

    def _compute_compact_draft_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        assert self.draft_window_size is not None
        visible_lens = torch.clamp(
            seq_lens.to(dtype=torch.int32, device=self.device),
            max=int(self.draft_window_size),
        )
        if self.page_size <= 1:
            return visible_lens

        # Paged FA backends derive the page table from local token positions, so the
        # compact suffix must start on a page boundary. Keep up to page_size - 1 extra
        # tokens on the left to preserve valid local page structure.
        seq_lens_i64 = seq_lens.to(torch.int64)
        visible_lens_i64 = visible_lens.to(torch.int64)
        visible_start = seq_lens_i64 - visible_lens_i64
        aligned_start = visible_start - torch.remainder(visible_start, self.page_size)
        return (seq_lens_i64 - aligned_start).to(torch.int32)

    def _resolve_mask_token_id(
        self, *, mask_token: str, mask_token_id: Optional[int] = None
    ) -> int:
        if not isinstance(mask_token, str) or not mask_token:
            raise ValueError(
                f"DFLASH mask_token must be a non-empty string, got {mask_token!r}."
            )

        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        if mask_token_id is not None:
            resolved_id = int(mask_token_id)
            if resolved_id >= vocab_size:
                raise ValueError(
                    "DFLASH mask_token_id is outside the target vocab size. "
                    f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                    f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                    "SGLang does not support resizing target embeddings for DFLASH yet."
                )

            tokenizer = getattr(self.target_worker, "tokenizer", None)
            if tokenizer is not None:
                token_id_from_vocab = tokenizer.get_vocab().get(mask_token, None)
                if (
                    token_id_from_vocab is not None
                    and int(token_id_from_vocab) != resolved_id
                ):
                    raise ValueError(
                        "DFLASH config mismatch: dflash_config.mask_token_id conflicts with tokenizer vocab id "
                        f"for dflash_config.mask_token. mask_token={mask_token!r}, "
                        f"mask_token_id={resolved_id}, tokenizer_vocab_id={int(token_id_from_vocab)}."
                    )
            return resolved_id

        tokenizer = getattr(self.target_worker, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "DFLASH requires tokenizer initialization when dflash_config.mask_token_id is not set "
                "(skip_tokenizer_init is not supported in this mode)."
            )

        resolved_id = None
        if getattr(tokenizer, "mask_token", None) == mask_token:
            resolved_id = getattr(tokenizer, "mask_token_id", None)

        if resolved_id is None:
            # Prefer checking the explicit vocab mapping first.
            vocab = tokenizer.get_vocab()
            resolved_id = vocab.get(mask_token, None)

        if resolved_id is None:
            # Mirror the reference DFlash HF demo by adding the mask token to the tokenizer.
            # This is safe only when the resulting id stays within the target model vocab size.
            added = tokenizer.add_special_tokens({"mask_token": mask_token})
            resolved_id = getattr(tokenizer, "mask_token_id", None)
            if resolved_id is None:
                resolved_id = tokenizer.convert_tokens_to_ids(mask_token)

            if added and self.tp_rank == 0:
                logger.info(
                    "Added DFLASH mask token to tokenizer. token=%s, mask_token_id=%s, tokenizer_len=%s, model_vocab_size=%s",
                    mask_token,
                    resolved_id,
                    len(tokenizer),
                    vocab_size,
                )

        if resolved_id is None or int(resolved_id) < 0:
            raise ValueError(
                "DFLASH requires resolving a mask token id, but it could not be resolved. "
                f"mask_token={mask_token!r}."
            )

        if resolved_id >= vocab_size:
            raise ValueError(
                "DFLASH mask_token_id is outside the target vocab size. "
                f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                "SGLang does not support resizing target embeddings for DFLASH yet."
            )

        return int(resolved_id)

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
    ):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return

        if batch.has_grammar:
            raise RuntimeError(
                "Invariant broken: DFLASH batch has grammar constraints, but scheduler should have rejected this request."
            )
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            if (
                not is_dflash_sampling_verify_available()
                and not self._warned_sampling_fallback
                and self.tp_rank == 0
            ):
                logger.warning(
                    "DFLASH non-greedy verification is unavailable on this build/device; "
                    "falling back to greedy argmax verification."
                )
                self._warned_sampling_fallback = True

        bs = batch.batch_size()

        # --- 1) Append any newly committed tokens into the draft KV cache.
        self._append_target_hidden_to_draft_kv(batch, draft_input)

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DFLASH requires the target model to expose a vocab-parallel `lm_head` with `weight` and "
                "`shard_indices` attributes."
            )

        # --- 2) Draft a non-causal block with the draft model.
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        block_ids.fill_(int(self._mask_token_id))
        block_ids[:, 0].copy_(draft_input.verified_id.to(torch.long))

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        # For spec-v1, the draft KV cache is always materialized before drafting the
        # next block. `target_prefix_lens` stay absolute for RoPE; `draft_prefix_lens`
        # are the logical resident lengths in the draft-local cache.
        target_prefix_lens = batch.seq_lens  # int32, device
        draft_prefix_lens = draft_input.draft_seq_lens
        if draft_prefix_lens.dtype != torch.int32:
            draft_prefix_lens = draft_prefix_lens.to(torch.int32)
        if draft_prefix_lens.device != self.device:
            draft_prefix_lens = draft_prefix_lens.to(self.device, non_blocking=True)

        positions_2d = self._draft_block_positions_buf[:bs]
        torch.add(
            target_prefix_lens.unsqueeze(1), self._block_pos_offsets, out=positions_2d
        )
        positions = positions_2d.reshape(-1)

        block_start = draft_prefix_lens
        block_end = self._draft_block_end_buf[:bs]
        torch.add(block_start, int(self.block_size), out=block_end)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))
        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        token_to_kv_pool_state_backup = allocator.backup_state()
        try:
            if self.page_size == 1:
                block_cache_loc = allocator.alloc(bs * self.block_size)
            else:
                block_end_cpu = seq_lens_cpu + int(self.block_size)
                last_loc = get_last_loc(
                    self.draft_model_runner.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    block_start,
                )
                block_cache_loc = allocator.alloc_extend(
                    block_start,
                    seq_lens_cpu,
                    block_end,
                    block_end_cpu,
                    last_loc,
                    bs * self.block_size,
                )
            if block_cache_loc is None:
                raise RuntimeError(
                    f"DFLASH draft OOM when allocating {bs * self.block_size} block tokens."
                )

            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                block_start,
                block_end,
                block_cache_loc,
                bs,
            )

            # Use TARGET_VERIFY mode (cuda-graphable) to run a fixed-size draft block.
            # In this mode, `seq_lens` stores the prefix lengths; attention backends
            # derive kv_len by adding `draft_token_num`.
            draft_spec_info = self._draft_block_spec_info
            seq_lens = draft_prefix_lens
            seq_lens_sum = int(draft_prefix_lens.sum().item())
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bs,
                input_ids=block_ids.flatten(),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=block_cache_loc,
                seq_lens_sum=seq_lens_sum,
                seq_lens_cpu=seq_lens_cpu,
                positions=positions,
                req_to_token_pool=self.draft_model_runner.req_to_token_pool,
                token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
                attn_backend=self.draft_model_runner.attn_backend,
                input_embeds=input_embeds,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                spec_info=draft_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            with torch.inference_mode():
                draft_logits_output = self.draft_model_runner.forward(
                    forward_batch
                ).logits_output
        finally:
            # Drop the speculative block from the shared allocator (EAGLE3-style).
            allocator.restore_state(token_to_kv_pool_state_backup)

        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFLASH draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, self.block_size, -1)

        if getattr(self.draft_model, "projector_type", None) in {"domino", "causal_v5"}:
            draft_next = self._v5_rollout_draft_block(
                draft_hidden=draft_hidden,
                verified_id=block_ids[:, 0],
                target_model=target_model,
                lm_head=lm_head,
            )
        else:
            draft_next = self._greedy_sample_from_vocab_parallel_head(
                hidden_states=draft_hidden[:, 1:, :].reshape(
                    -1, draft_hidden.shape[-1]
                ),
                lm_head=lm_head,
            ).view(bs, self.block_size - 1)
        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)
        positions = positions_2d.reshape(-1)

        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions,
            draft_token_num=self.block_size,
        )
        _, build_custom_mask = resolve_dflash_verify_mask_policy(
            self.model_runner.attn_backend
        )
        verify_input.prepare_for_verify(
            batch,
            self.page_size,
            build_custom_mask=build_custom_mask,
        )

        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def _greedy_sample_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Greedy argmax over the target LM head in a TP-safe way.

        We cannot materialize full logits for large vocabularies efficiently, and with
        TP>1 each rank only owns a shard of the LM head weight. This computes the
        per-rank max, gathers candidates across TP ranks, and selects the global max.
        """

        if hidden_states.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=hidden_states.device)

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        prof_enabled = os.environ.get("DFLASH_PROF_GREEDY") == "1" and hidden_states.is_cuda

        def prof_mark():
            if not prof_enabled:
                return None
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev

        evt_total_start = prof_mark()
        chunk_events = [] if prof_enabled else None

        if not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "DFLASH greedy sampling requires a vocab-parallel head with `weight` and `shard_indices`."
            )

        shard = lm_head.shard_indices
        weight = lm_head.weight  # [local_vocab_padded, hidden]
        weight_dtype = weight.dtype

        # Valid ranges in the local shard (excluding padding):
        #   base vocab:  [0, num_org)
        #   added vocab: [num_org_padded, num_org_padded + num_added)
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        num_tokens = int(hidden_states.shape[0])
        out_token_ids = torch.empty(
            (num_tokens,), dtype=torch.long, device=hidden_states.device
        )

        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        # Fast path (common): single-rank greedy sampling over the base vocab shard.
        # Avoids extra max/id bookkeeping that is only needed for TP sync or added vocab.
        if tp_size == 1 and num_added == 0:
            for start in range(0, num_tokens, int(chunk_size)):
                end = min(num_tokens, start + int(chunk_size))
                hs = _cast_hs(hidden_states[start:end])
                if num_org > 0:
                    base_logits = torch.matmul(hs, weight[:num_org].T)
                    out_token_ids[start:end] = (
                        torch.argmax(base_logits, dim=-1).to(torch.long)
                        + org_vocab_start
                    )
                else:
                    out_token_ids[start:end] = 0
            return out_token_ids

        for start in range(0, num_tokens, int(chunk_size)):
            end = min(num_tokens, start + int(chunk_size))
            hs = _cast_hs(hidden_states[start:end])
            chunk_len = int(hs.shape[0])
            evt_local_start = prof_mark()

            # Base vocab logits.
            if num_org > 0:
                base_logits = torch.matmul(hs, weight[:num_org].T)
                local_max, local_arg = torch.max(base_logits, dim=-1)
            else:
                local_max = torch.full(
                    (chunk_len,),
                    torch.finfo(weight_dtype).min,
                    dtype=weight_dtype,
                    device=hs.device,
                )
                local_arg = torch.zeros(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )

            # Added vocab logits (e.g., LoRA-added embeddings), if present.
            if num_added > 0:
                added_slice_start = num_org_padded
                added_slice_end = num_org_padded + num_added
                added_logits = torch.matmul(
                    hs, weight[added_slice_start:added_slice_end].T
                )
                added_max, added_arg = torch.max(added_logits, dim=-1)
                use_added = added_max > local_max
                local_max = torch.where(use_added, added_max, local_max)
                # For base/added conversion below, keep local_arg expressed in the full local
                # weight index space (base + padding + added), matching `lm_head.weight`.
                local_arg = torch.where(
                    use_added, added_arg.to(local_arg.dtype) + num_org_padded, local_arg
                )

            # Convert local argmax indices to global token ids.
            if num_added == 0:
                local_arg.add_(org_vocab_start)
                global_ids = local_arg
            else:
                global_ids = torch.empty(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )
                is_base = local_arg < num_org
                global_ids[is_base] = org_vocab_start + local_arg[is_base]
                global_ids[~is_base] = added_vocab_start + (
                    local_arg[~is_base] - num_org_padded
                )
            evt_local_end = prof_mark()

            if tp_size == 1:
                out_token_ids[start:end] = global_ids.to(torch.long)
                if prof_enabled:
                    chunk_events.append((evt_local_start, evt_local_end, evt_local_end))
                continue

            # Gather per-rank maxima and associated global ids, then select the global max.
            needed = tp_size * chunk_len
            chunk_cap = int(chunk_size)
            if (
                self._draft_greedy_gather_cap < needed
                or self._draft_greedy_gathered_max_buf is None
                or self._draft_greedy_gathered_ids_buf is None
                or self._draft_greedy_gathered_max_buf.dtype != local_max.dtype
                or self._draft_greedy_gathered_max_buf.device != hs.device
            ):
                # Allocate enough space for the max chunk size to avoid reallocations.
                cap = tp_size * chunk_cap
                self._draft_greedy_gathered_max_buf = torch.empty(
                    (cap,), dtype=local_max.dtype, device=hs.device
                )
                self._draft_greedy_gathered_ids_buf = torch.empty(
                    (cap,), dtype=global_ids.dtype, device=hs.device
                )
                self._draft_greedy_gather_cap = cap

            if (
                self._draft_greedy_index_cap < chunk_len
                or self._draft_greedy_best_rank_buf is None
                or self._draft_greedy_rank_index_buf is None
                or self._draft_greedy_selected_ids_buf is None
                or self._draft_greedy_best_rank_buf.device != hs.device
                or self._draft_greedy_selected_ids_buf.device != hs.device
            ):
                self._draft_greedy_best_rank_buf = torch.empty(
                    (chunk_cap,), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_rank_index_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_selected_ids_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_index_cap = chunk_cap

            gathered_max = self._draft_greedy_gathered_max_buf[:needed]
            gathered_ids = self._draft_greedy_gathered_ids_buf[:needed]

            tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
            tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
            gathered_max = gathered_max.view(tp_size, chunk_len)
            gathered_ids = gathered_ids.view(tp_size, chunk_len)

            best_rank = self._draft_greedy_best_rank_buf[:chunk_len]
            torch.argmax(gathered_max, dim=0, out=best_rank)

            rank_index = self._draft_greedy_rank_index_buf[:, :chunk_len]
            rank_index[0].copy_(best_rank)
            selected_ids = self._draft_greedy_selected_ids_buf[:, :chunk_len]
            torch.gather(gathered_ids, 0, rank_index, out=selected_ids)
            out_token_ids[start:end].copy_(selected_ids.view(-1))
            evt_reduce_end = prof_mark()
            if prof_enabled:
                chunk_events.append((evt_local_start, evt_local_end, evt_reduce_end))

        evt_total_end = prof_mark()
        if prof_enabled:
            torch.cuda.synchronize()

            def elapsed(start, end):
                if start is None or end is None:
                    return 0.0
                return start.elapsed_time(end)

            local_ms = 0.0
            reduce_ms = 0.0
            for local_start, local_end, reduce_end in chunk_events:
                local_ms += elapsed(local_start, local_end)
                reduce_ms += elapsed(local_end, reduce_end)
            total_ms = elapsed(evt_total_start, evt_total_end)
            prof = getattr(self, "_greedy_prof", None)
            if prof is None:
                prof = {
                    "n": 0,
                    "print_every": int(
                        os.environ.get("DFLASH_PROF_GREEDY_PRINT_EVERY", "20")
                    ),
                    "window": [],
                    "window_size": int(
                        os.environ.get("DFLASH_PROF_GREEDY_WINDOW", "20")
                    ),
                    "local_sum": 0.0,
                    "reduce_sum": 0.0,
                    "total_sum": 0.0,
                }
                self._greedy_prof = prof
            prof["n"] += 1
            prof["local_sum"] += local_ms
            prof["reduce_sum"] += reduce_ms
            prof["total_sum"] += total_ms
            prof["window"].append((local_ms, reduce_ms, total_ms))
            if len(prof["window"]) > prof["window_size"]:
                prof["window"].pop(0)
            if self.tp_rank == 0 and prof["n"] % max(prof["print_every"], 1) == 0:
                n = prof["n"]
                win = prof["window"]
                win_n = max(len(win), 1)
                logger.warning(
                    "[DFLASH greedy prof] rank=%d n_calls=%d tokens=%d "
                    "total=%.3fms(win %.3f) local=%.3f reduce=%.3f",
                    self.tp_rank,
                    n,
                    num_tokens,
                    prof["total_sum"] / n,
                    sum(item[2] for item in win) / win_n,
                    prof["local_sum"] / n,
                    prof["reduce_sum"] / n,
                )
        return out_token_ids

    def _global_argmax_from_local_logits(
        self,
        *,
        local_logits: torch.Tensor,
        local_vocab_start: int,
        local_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Select global token ids from per-rank vocab-shard logits."""

        if local_logits.ndim != 2:
            raise RuntimeError(
                "DFLASH global argmax expects [batch, local_vocab] logits, "
                f"got shape={tuple(local_logits.shape)}."
            )

        local_max, local_arg = torch.max(local_logits, dim=-1)
        if local_token_ids is None:
            global_ids = local_arg.to(torch.int64) + int(local_vocab_start)
        else:
            if tuple(local_token_ids.shape) != tuple(local_logits.shape):
                raise RuntimeError(
                    "DFLASH global argmax token-id shape mismatch. "
                    f"logits={tuple(local_logits.shape)}, ids={tuple(local_token_ids.shape)}."
                )
            global_ids = torch.gather(
                local_token_ids.to(torch.int64), 1, local_arg.unsqueeze(1)
            ).view(-1)
        return self._global_argmax_from_local_max(
            local_max=local_max,
            global_ids=global_ids,
        )

    def _global_argmax_from_local_max(
        self,
        *,
        local_max: torch.Tensor,
        global_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Select global token ids from per-rank local max values and ids."""

        if local_max.ndim != 1 or global_ids.ndim != 1:
            raise RuntimeError(
                "DFLASH global argmax expects 1D max/id tensors, "
                f"got local_max={tuple(local_max.shape)}, global_ids={tuple(global_ids.shape)}."
            )
        if int(local_max.shape[0]) != int(global_ids.shape[0]):
            raise RuntimeError(
                "DFLASH global argmax max/id length mismatch: "
                f"local_max={tuple(local_max.shape)}, global_ids={tuple(global_ids.shape)}."
            )

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        if tp_size == 1:
            return global_ids.to(torch.long)

        chunk_len = int(local_max.shape[0])

        pack_mode = os.environ.get("DFLASH_TP_PACK_ARGMAX", "1").lower()
        vocab_size = int(
            getattr(self.target_worker.model_runner.model_config, "vocab_size", 0) or 0
        )
        can_pack_ids = (
            pack_mode not in {"0", "false", "off", "disable", "disabled"}
            and 0 < vocab_size < (1 << 24)
        )
        if can_pack_ids:
            # Pack score and token id into a tiny fp32 tensor so TP argmax needs
            # one all-gather instead of separate score/id collectives. Token ids
            # below 2^24 are exactly representable in fp32.
            if (
                self._draft_greedy_pair_cap != chunk_len
                or self._draft_greedy_pair_tp_size != tp_size
                or self._draft_greedy_local_pair_buf is None
                or self._draft_greedy_gathered_pair_buf is None
                or self._draft_greedy_selected_ids_f32_buf is None
                or self._draft_greedy_local_pair_buf.device != local_max.device
            ):
                self._draft_greedy_local_pair_buf = torch.empty(
                    (2, chunk_len), dtype=torch.float32, device=local_max.device
                )
                self._draft_greedy_gathered_pair_buf = torch.empty(
                    (tp_size, 2, chunk_len),
                    dtype=torch.float32,
                    device=local_max.device,
                )
                self._draft_greedy_selected_ids_f32_buf = torch.empty(
                    (1, chunk_len), dtype=torch.float32, device=local_max.device
                )
                self._draft_greedy_pair_cap = chunk_len
                self._draft_greedy_pair_tp_size = tp_size

            local_pair = self._draft_greedy_local_pair_buf
            gathered_pair = self._draft_greedy_gathered_pair_buf
            assert local_pair is not None
            assert gathered_pair is not None
            local_pair[0].copy_(local_max.float())
            local_pair[1].copy_(global_ids.to(torch.float32))
            tp_group.all_gather_into_tensor(gathered_pair, local_pair)

            if (
                self._draft_greedy_index_cap < chunk_len
                or self._draft_greedy_best_rank_buf is None
                or self._draft_greedy_rank_index_buf is None
                or self._draft_greedy_best_rank_buf.device != local_max.device
            ):
                self._draft_greedy_best_rank_buf = torch.empty(
                    (chunk_len,), dtype=torch.int64, device=local_max.device
                )
                self._draft_greedy_rank_index_buf = torch.empty(
                    (1, chunk_len), dtype=torch.int64, device=local_max.device
                )
                self._draft_greedy_index_cap = chunk_len

            gathered_max = gathered_pair[:, 0, :]
            gathered_ids = gathered_pair[:, 1, :]
            best_rank = self._draft_greedy_best_rank_buf[:chunk_len]
            torch.argmax(gathered_max, dim=0, out=best_rank)

            rank_index = self._draft_greedy_rank_index_buf[:, :chunk_len]
            rank_index[0].copy_(best_rank)
            selected_ids_f32 = self._draft_greedy_selected_ids_f32_buf
            assert selected_ids_f32 is not None
            torch.gather(gathered_ids, 0, rank_index, out=selected_ids_f32)
            return selected_ids_f32.view(-1).to(torch.long)

        needed = tp_size * chunk_len
        if (
            self._draft_greedy_gather_cap < needed
            or self._draft_greedy_gathered_max_buf is None
            or self._draft_greedy_gathered_ids_buf is None
            or self._draft_greedy_gathered_max_buf.dtype != local_max.dtype
            or self._draft_greedy_gathered_max_buf.device != local_max.device
        ):
            self._draft_greedy_gathered_max_buf = torch.empty(
                (needed,), dtype=local_max.dtype, device=local_max.device
            )
            self._draft_greedy_gathered_ids_buf = torch.empty(
                (needed,), dtype=global_ids.dtype, device=local_max.device
            )
            self._draft_greedy_gather_cap = needed

        if (
            self._draft_greedy_index_cap < chunk_len
            or self._draft_greedy_best_rank_buf is None
            or self._draft_greedy_rank_index_buf is None
            or self._draft_greedy_selected_ids_buf is None
            or self._draft_greedy_best_rank_buf.device != local_max.device
            or self._draft_greedy_selected_ids_buf.device != local_max.device
        ):
            self._draft_greedy_best_rank_buf = torch.empty(
                (chunk_len,), dtype=torch.int64, device=local_max.device
            )
            self._draft_greedy_rank_index_buf = torch.empty(
                (1, chunk_len), dtype=torch.int64, device=local_max.device
            )
            self._draft_greedy_selected_ids_buf = torch.empty(
                (1, chunk_len), dtype=torch.int64, device=local_max.device
            )
            self._draft_greedy_index_cap = chunk_len

        gathered_max = self._draft_greedy_gathered_max_buf[:needed]
        gathered_ids = self._draft_greedy_gathered_ids_buf[:needed]
        tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
        tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())

        gathered_max = gathered_max.view(tp_size, chunk_len)
        gathered_ids = gathered_ids.view(tp_size, chunk_len)
        best_rank = self._draft_greedy_best_rank_buf[:chunk_len]
        torch.argmax(gathered_max, dim=0, out=best_rank)

        rank_index = self._draft_greedy_rank_index_buf[:, :chunk_len]
        rank_index[0].copy_(best_rank)
        selected_ids = self._draft_greedy_selected_ids_buf[:, :chunk_len]
        torch.gather(gathered_ids, 0, rank_index, out=selected_ids)
        return selected_ids.view(-1).to(torch.long)

    def _v5_rollout_draft_block_tp_eager(
        self,
        *,
        z: torch.Tensor,
        verified_id: torch.Tensor,
        target_model,
        lm_head,
        org_vocab_start: int,
        num_org: int,
        num_org_padded: int,
        state: dict,
    ) -> torch.Tensor:
        """Domino rollout for TP>1.

        Each rank scores its local vocab shard, then synchronizes the winning
        token after every step because the selected token feeds the next GRU
        state. This path keeps the TP collectives explicit while using fused
        local scoring and table-based GRU input lookup when available.
        """

        bs, num_draft, hidden_size = z.shape
        device = z.device
        weight = lm_head.weight
        weight_dtype = weight.dtype
        embed_module = target_model.get_input_embeddings()
        draft_model = self.draft_model
        valid_vocab_size = int(
            getattr(lm_head, "org_vocab_size", 0)
            or getattr(self.target_worker.model_runner.model_config, "vocab_size", 0)
            or int(state["fc2_weight"].shape[0])
        )
        if valid_vocab_size <= 0:
            raise RuntimeError(
                f"DFLASH TP replicated scorer cannot resolve valid vocab size: {valid_vocab_size}."
            )
        if valid_vocab_size > int(state["fc2_weight"].shape[0]):
            raise RuntimeError(
                "DFLASH Domino fc2 vocab is smaller than the target org vocab: "
                f"fc2_vocab={int(state['fc2_weight'].shape[0])}, "
                f"target_vocab={valid_vocab_size}."
            )
        full_lm_head_weight = self._get_v5_tp_full_lm_head_weight(
            local_lm_head_weight=weight,
            org_vocab_start=org_vocab_start,
            num_org_padded=num_org_padded,
            valid_vocab_size=valid_vocab_size,
        )
        use_replicated_scorer = full_lm_head_weight is not None
        prof_enabled = os.environ.get("DFLASH_PROF_V5_TP") == "1" and z.is_cuda

        def prof_mark():
            if not prof_enabled:
                return None
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev

        evt_total_start = prof_mark()
        evt_table_start = prof_mark()
        gru_input_table = draft_model.get_v5_gru_input_proj_table(embed_module.weight)
        full_gru_input_table = self._get_v5_tp_full_gru_input_table(
            local_gru_input_table=gru_input_table,
            org_vocab_start=org_vocab_start,
            num_org_padded=num_org_padded,
        )
        evt_table_end = prof_mark()

        z_for_dtype = z.to(weight_dtype) if z.dtype != weight_dtype else z
        score_weight = full_lm_head_weight if use_replicated_scorer else weight[:num_org]
        score_vocab_size = valid_vocab_size if use_replicated_scorer else num_org
        evt_base_start = prof_mark()
        base_logits = torch.matmul(
            z_for_dtype.reshape(bs * num_draft, hidden_size),
            score_weight.T,
        ).view(bs, num_draft, score_vocab_size)
        evt_base_end = prof_mark()

        out = torch.empty((bs, num_draft), dtype=torch.long, device=device)
        evt_slot1_start = prof_mark()
        if use_replicated_scorer:
            slot_1 = torch.argmax(base_logits[:, 0, :], dim=-1).to(torch.long)
        else:
            slot_1 = self._global_argmax_from_local_logits(
                local_logits=base_logits[:, 0, :],
                local_vocab_start=org_vocab_start,
            )
        evt_slot1_end = prof_mark()
        out[:, 0].copy_(slot_1)

        prefix_ids = torch.stack([verified_id.to(torch.long), slot_1], dim=1)
        evt_init_start = prof_mark()
        if full_gru_input_table is not None:
            prefix_gi = self._v5_lookup_gru_input_proj_full(
                token_ids=prefix_ids,
                full_gru_input_table=full_gru_input_table,
            )
        else:
            prefix_gi = self._v5_lookup_gru_input_proj_tp(
                token_ids=prefix_ids,
                local_gru_input_table=gru_input_table,
                org_vocab_start=org_vocab_start,
                num_org=num_org,
            )
        gru_h = torch.zeros(
            (bs, int(state["gru_hidden_size"])),
            dtype=prefix_gi.dtype,
            device=device,
        )
        for t_idx in range(int(prefix_ids.shape[1])):
            gru_h = self._v5_manual_gru_step_from_input_proj(
                gi=prefix_gi[:, t_idx, :],
                gru_h=gru_h,
                state=state,
            )
        evt_init_end = prof_mark()

        evt_zproj_start = prof_mark()
        z_proj_all = torch.nn.functional.linear(z, state["w_z"], state["b1"])
        evt_zproj_end = prof_mark()
        if use_replicated_scorer:
            fc2_w = state["fc2_weight"][:valid_vocab_size].contiguous()
            fc2_b = (
                state["fc2_bias"][:valid_vocab_size].contiguous()
                if state["fc2_bias"] is not None
                else None
            )
        else:
            fc2_w = state["fc2_weight"][
                org_vocab_start : org_vocab_start + num_org
            ].contiguous()
            fc2_b = (
                state["fc2_bias"][
                    org_vocab_start : org_vocab_start + num_org
                ].contiguous()
                if state["fc2_bias"] is not None
                else None
            )
        candidate_pool_size = max(
            0, int(os.environ.get("DFLASH_V5_CANDIDATE_POOL", "1024"))
        )
        if candidate_pool_size > 0 and candidate_pool_size >= score_vocab_size:
            candidate_pool_size = 0
        fused_score_mode = os.environ.get("DFLASH_V5_TP_FUSED_SCORE", "1").lower()
        use_fused_score = z.is_cuda and fused_score_mode not in {
            "0",
            "false",
            "off",
            "disable",
            "disabled",
        }
        candidate_ids = None
        candidate_token_ids = None
        candidate_fc2_w = None
        candidate_fc2_b = None
        evt_candidate_start = prof_mark()
        if candidate_pool_size > 0 and num_draft > 1:
            pool_source = base_logits[:, 1:, :]
            pool_logits = pool_source.max(dim=1).values
            candidate_ids = torch.topk(
                pool_logits, k=candidate_pool_size, dim=-1
            ).indices.contiguous()
            if not use_fused_score:
                candidate_token_ids = candidate_ids.to(torch.int64)
                if not use_replicated_scorer:
                    candidate_token_ids = candidate_token_ids + int(org_vocab_start)
                candidate_fc2_w = fc2_w.index_select(0, candidate_ids.reshape(-1)).view(
                    bs, candidate_pool_size, -1
                )
                if fc2_b is not None:
                    candidate_fc2_b = fc2_b.index_select(
                        0, candidate_ids.reshape(-1)
                    ).view(bs, candidate_pool_size)
        evt_candidate_end = prof_mark()

        if use_fused_score:
            block_v = int(os.environ.get("DFLASH_V5_TP_BLOCK_V", "512"))
            block_m = int(os.environ.get("DFLASH_V5_TP_BLOCK_M", "32"))
            score_num_warps = int(os.environ.get("DFLASH_V5_TP_SCORE_NUM_WARPS", "4"))
            score_num_stages = int(os.environ.get("DFLASH_V5_TP_SCORE_NUM_STAGES", "3"))
            candidate_block_c = int(
                os.environ.get("DFLASH_V5_TP_CANDIDATE_BLOCK_C", "512")
            )
            num_score_blocks = (
                (candidate_pool_size + candidate_block_c - 1) // candidate_block_c
                if candidate_ids is not None
                else (score_vocab_size + block_v - 1) // block_v
            )
            score_val_buf = torch.empty(
                (bs, num_score_blocks), dtype=torch.float32, device=device
            )
            score_idx_buf = torch.empty(
                (bs, num_score_blocks), dtype=torch.int32, device=device
            )
            local_tok_buf = torch.empty((bs,), dtype=torch.long, device=device)
            local_val_buf = (
                None
                if use_replicated_scorer
                else torch.empty((bs,), dtype=torch.float32, device=device)
            )

        G = int(state["gru_hidden_size"])
        emb_dim = int(state["w_z"].shape[0])
        w_s_hh_T = state.get("w_s_hh_T", None)
        evt_loop_start = prof_mark()
        step_events = [] if prof_enabled else None
        for k in range(1, num_draft):
            evt_step_start = prof_mark()
            gh = None
            if w_s_hh_T is not None and k + 1 < num_draft:
                sh = torch.matmul(gru_h, w_s_hh_T)
                s_proj = sh[:, :emb_dim]
                gh = sh[:, emb_dim : emb_dim + 3 * G]
            else:
                s_proj = torch.nn.functional.linear(gru_h, state["w_s"], None)
            evt_proj_end = prof_mark()
            if use_fused_score:
                if candidate_ids is None:
                    if use_replicated_scorer:
                        fused_silu_fc2_argmax(
                            z_proj=z_proj_all[:, k, :],
                            s_proj=s_proj,
                            fc2_weight=fc2_w,
                            fc2_bias=fc2_b,
                            base_logits=base_logits[:, k, :],
                            out_val=score_val_buf,
                            out_idx=score_idx_buf,
                            final_token=local_tok_buf,
                            block_v=block_v,
                            block_m=block_m,
                            num_warps=score_num_warps,
                            num_stages=score_num_stages,
                        )
                    else:
                        assert local_val_buf is not None
                        fused_silu_fc2_argmax_with_value(
                            z_proj=z_proj_all[:, k, :],
                            s_proj=s_proj,
                            fc2_weight=fc2_w,
                            fc2_bias=fc2_b,
                            base_logits=base_logits[:, k, :],
                            out_val=score_val_buf,
                            out_idx=score_idx_buf,
                            final_token=local_tok_buf,
                            final_value=local_val_buf,
                            block_v=block_v,
                            block_m=block_m,
                            num_warps=score_num_warps,
                            num_stages=score_num_stages,
                        )
                else:
                    if use_replicated_scorer:
                        fused_silu_fc2_candidate_argmax(
                            z_proj=z_proj_all[:, k, :],
                            s_proj=s_proj,
                            fc2_weight=fc2_w,
                            fc2_bias=fc2_b,
                            base_logits=base_logits[:, k, :],
                            candidate_ids=candidate_ids,
                            out_val=score_val_buf,
                            out_idx=score_idx_buf,
                            final_token=local_tok_buf,
                            block_c=candidate_block_c,
                            block_m=block_m,
                            num_warps=score_num_warps,
                            num_stages=score_num_stages,
                        )
                    else:
                        assert local_val_buf is not None
                        fused_silu_fc2_candidate_argmax_with_value(
                            z_proj=z_proj_all[:, k, :],
                            s_proj=s_proj,
                            fc2_weight=fc2_w,
                            fc2_bias=fc2_b,
                            base_logits=base_logits[:, k, :],
                            candidate_ids=candidate_ids,
                            out_val=score_val_buf,
                            out_idx=score_idx_buf,
                            final_token=local_tok_buf,
                            final_value=local_val_buf,
                            block_c=candidate_block_c,
                            block_m=block_m,
                            num_warps=score_num_warps,
                            num_stages=score_num_stages,
                        )
                evt_score_end = prof_mark()
                if use_replicated_scorer:
                    tok = local_tok_buf.to(torch.long)
                else:
                    tok = self._global_argmax_from_local_max(
                        local_max=local_val_buf,
                        global_ids=local_tok_buf.to(torch.int64) + int(org_vocab_start),
                    )
            elif candidate_ids is None:
                mid = torch.nn.functional.silu(z_proj_all[:, k, :] + s_proj)
                bias_local = torch.nn.functional.linear(mid, fc2_w, fc2_b)
                logits_k = base_logits[:, k, :] + bias_local.to(base_logits.dtype)
                evt_score_end = prof_mark()
                if use_replicated_scorer:
                    tok = torch.argmax(logits_k, dim=-1).to(torch.long)
                else:
                    tok = self._global_argmax_from_local_logits(
                        local_logits=logits_k,
                        local_vocab_start=org_vocab_start,
                    )
            else:
                assert candidate_fc2_w is not None
                assert candidate_token_ids is not None
                mid = torch.nn.functional.silu(z_proj_all[:, k, :] + s_proj)
                bias_local = torch.bmm(
                    candidate_fc2_w, mid.unsqueeze(-1)
                ).squeeze(-1)
                if candidate_fc2_b is not None:
                    bias_local = bias_local + candidate_fc2_b
                logits_k = (
                    torch.gather(base_logits[:, k, :], 1, candidate_ids)
                    + bias_local.to(base_logits.dtype)
                )
                evt_score_end = prof_mark()
                if use_replicated_scorer:
                    local_arg = torch.argmax(logits_k, dim=-1)
                    tok = torch.gather(
                        candidate_token_ids, 1, local_arg.unsqueeze(1)
                    ).view(-1)
                else:
                    tok = self._global_argmax_from_local_logits(
                        local_logits=logits_k,
                        local_vocab_start=org_vocab_start,
                        local_token_ids=candidate_token_ids,
                    )
            evt_reduce_end = prof_mark()
            out[:, k].copy_(tok)
            if k + 1 < num_draft:
                if full_gru_input_table is not None:
                    gi = self._v5_lookup_gru_input_proj_full(
                        token_ids=tok,
                        full_gru_input_table=full_gru_input_table,
                    )
                else:
                    gi = self._v5_lookup_gru_input_proj_tp(
                        token_ids=tok,
                        local_gru_input_table=gru_input_table,
                        org_vocab_start=org_vocab_start,
                        num_org=num_org,
                    )
                if gh is None:
                    gru_h = self._v5_manual_gru_step_from_input_proj(
                        gi=gi,
                        gru_h=gru_h,
                        state=state,
                    )
                else:
                    gru_h = self._v5_manual_gru_step_from_projections(
                        gi=gi,
                        gh=gh,
                        gru_h=gru_h,
                        state=state,
                    )
            evt_gru_end = prof_mark()
            if prof_enabled:
                step_events.append(
                    (
                        evt_step_start,
                        evt_proj_end,
                        evt_score_end,
                        evt_reduce_end,
                        evt_gru_end,
                    )
                )

        evt_loop_end = prof_mark()
        evt_total_end = prof_mark()
        if prof_enabled:
            torch.cuda.synchronize()

            def elapsed(start, end):
                if start is None or end is None:
                    return 0.0
                return start.elapsed_time(end)

            step_proj_ms = 0.0
            step_score_ms = 0.0
            step_reduce_ms = 0.0
            step_gru_ms = 0.0
            for s0, s1, s2, s3, s4 in step_events:
                step_proj_ms += elapsed(s0, s1)
                step_score_ms += elapsed(s1, s2)
                step_reduce_ms += elapsed(s2, s3)
                step_gru_ms += elapsed(s3, s4)

            values = {
                "table_ms": elapsed(evt_table_start, evt_table_end),
                "base_ms": elapsed(evt_base_start, evt_base_end),
                "slot1_ms": elapsed(evt_slot1_start, evt_slot1_end),
                "init_ms": elapsed(evt_init_start, evt_init_end),
                "zproj_ms": elapsed(evt_zproj_start, evt_zproj_end),
                "candidate_ms": elapsed(evt_candidate_start, evt_candidate_end),
                "loop_ms": elapsed(evt_loop_start, evt_loop_end),
                "step_proj_ms": step_proj_ms,
                "step_score_ms": step_score_ms,
                "step_reduce_ms": step_reduce_ms,
                "step_gru_ms": step_gru_ms,
                "total_ms": elapsed(evt_total_start, evt_total_end),
            }
            prof = getattr(self, "_v5_tp_prof", None)
            if prof is None:
                prof = {
                    "n": 0,
                    "print_every": int(
                        os.environ.get("DFLASH_PROF_V5_TP_PRINT_EVERY", "20")
                    ),
                    "sums": {k: 0.0 for k in values},
                    "window": [],
                    "window_size": int(
                        os.environ.get("DFLASH_PROF_V5_TP_WINDOW", "20")
                    ),
                }
                self._v5_tp_prof = prof
            prof["n"] += 1
            for key, value in values.items():
                prof["sums"][key] += value
            prof["window"].append(values)
            if len(prof["window"]) > prof["window_size"]:
                prof["window"].pop(0)
            if self.tp_rank == 0 and prof["n"] % max(prof["print_every"], 1) == 0:
                n = prof["n"]
                win = prof["window"]

                def avg(key):
                    return prof["sums"][key] / n

                def win_avg(key):
                    return sum(item[key] for item in win) / max(len(win), 1)

                logger.warning(
                    "[DFLASH v5 TP prof] rank=%d bs=%d n_calls=%d "
                    "scorer=%s total=%.3fms(win %.3f) table=%.3f base=%.3f slot1=%.3f "
                    "init=%.3f zproj=%.3f candidate=%.3f loop=%.3f(win %.3f) "
                    "loop_parts: proj=%.3f score=%.3f reduce=%.3f gru=%.3f",
                    self.tp_rank,
                    bs,
                    n,
                    "replicated" if use_replicated_scorer else "tp_reduce",
                    avg("total_ms"),
                    win_avg("total_ms"),
                    avg("table_ms"),
                    avg("base_ms"),
                    avg("slot1_ms"),
                    avg("init_ms"),
                    avg("zproj_ms"),
                    avg("candidate_ms"),
                    avg("loop_ms"),
                    win_avg("loop_ms"),
                    avg("step_proj_ms"),
                    avg("step_score_ms"),
                    avg("step_reduce_ms"),
                    avg("step_gru_ms"),
                )

        return out

    def _get_v5_tp_full_lm_head_weight(
        self,
        *,
        local_lm_head_weight: torch.Tensor,
        org_vocab_start: int,
        num_org_padded: int,
        valid_vocab_size: int,
    ) -> Optional[torch.Tensor]:
        """Replicate the target lm_head org-vocab shards for Domino TP rollout.

        Domino's GRU state depends on the exact token sampled at each draft step.
        If the scorer remains vocab-parallel, every step needs a TP global argmax
        before the next GRU step can run. Replicating the full org-vocab lm_head
        lets each rank compute the same winning token locally and removes that
        per-step collective. This is an optimization-only path and can be disabled
        with DFLASH_V5_TP_REPLICATE_SCORER=0.
        """

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        if tp_size == 1:
            return local_lm_head_weight[:valid_vocab_size]

        mode = os.environ.get("DFLASH_V5_TP_REPLICATE_SCORER", "auto").lower()
        if mode in {"0", "false", "off", "disable", "disabled"}:
            return None
        if mode not in {"1", "true", "on", "enable", "enabled", "auto"}:
            if not getattr(self, "_v5_tp_replicate_scorer_bad_mode_warned", False):
                logger.warning(
                    "Ignoring invalid DFLASH_V5_TP_REPLICATE_SCORER=%r; "
                    "expected auto/1/0. Falling back to auto.",
                    mode,
                )
                self._v5_tp_replicate_scorer_bad_mode_warned = True
            mode = "auto"

        if num_org_padded <= 0:
            return None
        expected_org_start = int(tp_group.rank_in_group) * int(num_org_padded)
        if int(org_vocab_start) != expected_org_start:
            if not getattr(self, "_v5_tp_replicate_scorer_layout_warned", False):
                logger.warning(
                    "DFLASH TP replicated scorer disabled because vocab shard layout "
                    "is not direct padded-org concat: org_vocab_start=%d, expected=%d.",
                    int(org_vocab_start),
                    expected_org_start,
                )
                self._v5_tp_replicate_scorer_layout_warned = True
            return None

        full_padded_vocab_size = tp_size * int(num_org_padded)
        if int(valid_vocab_size) > full_padded_vocab_size:
            raise RuntimeError(
                "DFLASH TP replicated scorer valid vocab exceeds padded org vocab: "
                f"valid_vocab_size={int(valid_vocab_size)}, "
                f"full_padded_vocab_size={full_padded_vocab_size}."
            )
        if int(local_lm_head_weight.shape[0]) < int(num_org_padded):
            raise RuntimeError(
                "DFLASH TP lm_head shard is smaller than the padded org-vocab shard: "
                f"weight_rows={int(local_lm_head_weight.shape[0])}, "
                f"num_org_padded={int(num_org_padded)}."
            )

        hidden_size = int(local_lm_head_weight.shape[-1])
        local_org_weight = local_lm_head_weight[: int(num_org_padded)].contiguous()
        cache_key = (
            local_lm_head_weight.data_ptr(),
            int(num_org_padded),
            int(valid_vocab_size),
            hidden_size,
            local_lm_head_weight.dtype,
            local_lm_head_weight.device,
            tp_size,
        )
        cached = getattr(self, "_v5_tp_full_lm_head_weight_cache", None)
        if cached is not None and cached.get("key") == cache_key:
            return cached["weight"]

        required_bytes = (
            tp_size
            * int(num_org_padded)
            * hidden_size
            * local_lm_head_weight.element_size()
        )
        if mode == "auto" and local_lm_head_weight.device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(local_lm_head_weight.device)
            cushion = int(os.environ.get("DFLASH_V5_TP_SCORER_CUSHION_MB", "2048"))
            if int(free_bytes) < required_bytes + cushion * 1024 * 1024:
                if not getattr(self, "_v5_tp_replicate_scorer_mem_warned", False):
                    logger.warning(
                        "DFLASH TP replicated scorer disabled by auto memory check: "
                        "need %.2f GiB + %d MiB cushion, free %.2f GiB. "
                        "Set DFLASH_V5_TP_REPLICATE_SCORER=1 to force or 0 to silence.",
                        required_bytes / (1024**3),
                        cushion,
                        int(free_bytes) / (1024**3),
                    )
                    self._v5_tp_replicate_scorer_mem_warned = True
                return None

        full_padded_weight = torch.empty(
            (full_padded_vocab_size, hidden_size),
            dtype=local_lm_head_weight.dtype,
            device=local_lm_head_weight.device,
        )
        tp_group.all_gather_into_tensor(full_padded_weight, local_org_weight)
        full_weight = full_padded_weight[: int(valid_vocab_size)]
        self._v5_tp_full_lm_head_weight_cache = {
            "key": cache_key,
            "weight": full_weight,
            "full_padded_weight": full_padded_weight,
        }
        if not getattr(self, "_v5_tp_replicate_scorer_logged", False):
            logger.info(
                "DFLASH TP replicated scorer enabled. lm_head_shape=%s, memory=%.2f GiB",
                tuple(full_weight.shape),
                required_bytes / (1024**3),
            )
            self._v5_tp_replicate_scorer_logged = True
        return full_weight

    def _get_v5_tp_full_gru_input_table(
        self,
        *,
        local_gru_input_table: torch.Tensor,
        org_vocab_start: int,
        num_org_padded: int,
    ) -> Optional[torch.Tensor]:
        """Replicate the padded org-vocab GRU input table across TP ranks.

        In the TP eager rollout the selected draft token is global, so the next
        GRU input projection currently requires one all-reduce per generated
        draft token. Gathering the padded org-vocab table once lets every rank
        index by global token id directly. The gathered layout is TP-concatenated
        padded org shards, so valid base token ids map to the same row id.
        """

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        if tp_size == 1:
            return local_gru_input_table

        mode = os.environ.get("DFLASH_V5_TP_REPLICATE_GRU_TABLE", "auto").lower()
        if mode in {"0", "false", "off", "disable", "disabled"}:
            return None
        if mode not in {"1", "true", "on", "enable", "enabled", "auto"}:
            if not getattr(self, "_v5_tp_replicate_gru_table_bad_mode_warned", False):
                logger.warning(
                    "Ignoring invalid DFLASH_V5_TP_REPLICATE_GRU_TABLE=%r; "
                    "expected auto/1/0. Falling back to auto.",
                    mode,
                )
                self._v5_tp_replicate_gru_table_bad_mode_warned = True
            mode = "auto"

        if num_org_padded <= 0:
            return None
        expected_org_start = int(tp_group.rank_in_group) * int(num_org_padded)
        if int(org_vocab_start) != expected_org_start:
            if not getattr(self, "_v5_tp_replicate_gru_table_layout_warned", False):
                logger.warning(
                    "DFLASH TP replicated GRU table disabled because vocab shard layout "
                    "is not direct padded-org concat: org_vocab_start=%d, expected=%d.",
                    int(org_vocab_start),
                    expected_org_start,
                )
                self._v5_tp_replicate_gru_table_layout_warned = True
            return None
        if int(local_gru_input_table.shape[0]) < int(num_org_padded):
            raise RuntimeError(
                "DFLASH TP GRU input table is smaller than the padded org-vocab shard: "
                f"table_rows={int(local_gru_input_table.shape[0])}, "
                f"num_org_padded={int(num_org_padded)}."
            )

        width = int(local_gru_input_table.shape[-1])
        local_org_table = local_gru_input_table[: int(num_org_padded)].contiguous()
        cache_key = (
            local_gru_input_table.data_ptr(),
            int(num_org_padded),
            width,
            local_gru_input_table.dtype,
            local_gru_input_table.device,
            tp_size,
        )
        cached = getattr(self, "_v5_tp_full_gru_input_table_cache", None)
        if cached is not None and cached.get("key") == cache_key:
            return cached["table"]

        required_bytes = (
            tp_size
            * int(num_org_padded)
            * width
            * local_gru_input_table.element_size()
        )
        if mode == "auto" and local_gru_input_table.device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(local_gru_input_table.device)
            # Keep a modest cushion for verify/draft temporary buffers. The table
            # is an optimization, so do not risk turning a viable run into OOM.
            cushion = int(os.environ.get("DFLASH_V5_TP_GRU_TABLE_CUSHION_MB", "1024"))
            if int(free_bytes) < required_bytes + cushion * 1024 * 1024:
                if not getattr(self, "_v5_tp_replicate_gru_table_mem_warned", False):
                    logger.warning(
                        "DFLASH TP replicated GRU table disabled by auto memory check: "
                        "need %.2f GiB + %d MiB cushion, free %.2f GiB. "
                        "Set DFLASH_V5_TP_REPLICATE_GRU_TABLE=1 to force or 0 to silence.",
                        required_bytes / (1024**3),
                        cushion,
                        int(free_bytes) / (1024**3),
                    )
                    self._v5_tp_replicate_gru_table_mem_warned = True
                return None

        full_table = torch.empty(
            (tp_size * int(num_org_padded), width),
            dtype=local_gru_input_table.dtype,
            device=local_gru_input_table.device,
        )
        tp_group.all_gather_into_tensor(full_table, local_org_table)
        self._v5_tp_full_gru_input_table_cache = {
            "key": cache_key,
            "table": full_table,
        }
        if not getattr(self, "_v5_tp_replicate_gru_table_logged", False):
            logger.info(
                "DFLASH TP replicated GRU input table enabled. shape=%s, memory=%.2f GiB",
                tuple(full_table.shape),
                required_bytes / (1024**3),
            )
            self._v5_tp_replicate_gru_table_logged = True
        return full_table

    def _v5_lookup_gru_input_proj_full(
        self,
        *,
        token_ids: torch.Tensor,
        full_gru_input_table: torch.Tensor,
    ) -> torch.Tensor:
        flat_ids = token_ids.to(torch.long).reshape(-1)
        gi = torch.index_select(full_gru_input_table, 0, flat_ids).contiguous()
        return gi.view(*token_ids.shape, full_gru_input_table.shape[-1])

    def _v5_lookup_gru_input_proj_tp(
        self,
        *,
        token_ids: torch.Tensor,
        local_gru_input_table: torch.Tensor,
        org_vocab_start: int,
        num_org: int,
    ) -> torch.Tensor:
        """Lookup token GRU input projections from vocab shards and all-reduce."""

        if num_org <= 0:
            raise RuntimeError("DFLASH TP GRU input lookup got an empty vocab shard.")

        flat_ids = token_ids.to(torch.long).reshape(-1)
        in_local = (flat_ids >= int(org_vocab_start)) & (
            flat_ids < int(org_vocab_start + num_org)
        )
        local_idx = (flat_ids - int(org_vocab_start)).clamp_(0, int(num_org) - 1)
        local_gi = torch.index_select(
            local_gru_input_table, 0, local_idx
        ).contiguous()
        local_gi.masked_fill_(~in_local.unsqueeze(-1), 0)
        global_gi = get_tp_group().all_reduce(local_gi)
        return global_gi.view(*token_ids.shape, local_gru_input_table.shape[-1])

    def _v5_manual_gru_step_from_input_proj(
        self,
        *,
        gi: torch.Tensor,
        gru_h: torch.Tensor,
        state: dict,
    ) -> torch.Tensor:
        """Run one GRU step from precomputed input projection gates."""

        G = int(state["gru_hidden_size"])
        gh = torch.nn.functional.linear(gru_h, state["w_hh"], state["b_hh"])
        r = torch.sigmoid(gi[:, :G] + gh[:, :G])
        z_gate = torch.sigmoid(gi[:, G : 2 * G] + gh[:, G : 2 * G])
        n = torch.tanh(gi[:, 2 * G :] + r * gh[:, 2 * G :])
        return (1.0 - z_gate) * n + z_gate * gru_h

    def _v5_manual_gru_step_from_projections(
        self,
        *,
        gi: torch.Tensor,
        gh: torch.Tensor,
        gru_h: torch.Tensor,
        state: dict,
    ) -> torch.Tensor:
        """Run one GRU step when both input and hidden projections are ready."""

        if state["b_hh"] is not None:
            gh = gh + state["b_hh"]
        G = int(state["gru_hidden_size"])
        r = torch.sigmoid(gi[:, :G] + gh[:, :G])
        z_gate = torch.sigmoid(gi[:, G : 2 * G] + gh[:, G : 2 * G])
        n = torch.tanh(gi[:, 2 * G :] + r * gh[:, 2 * G :])
        return (1.0 - z_gate) * n + z_gate * gru_h

    def _accumulate_and_print_breakdown(self, breakdown: dict) -> None:
        """Accumulate per-step loop timings and print summary every N calls."""
        acc = getattr(self, "_v5_breakdown_acc", None)
        if acc is None:
            acc = {
                "n": 0,
                "print_every": 10,
                "gemm": [0.0] * len(breakdown["gemm"]),
                "score": [0.0] * len(breakdown["score"]),
                "score_main": [0.0] * len(breakdown.get("score_main", [])),
                "score_reduce": [0.0] * len(breakdown.get("score_reduce", [])),
                "gru": [0.0] * len(breakdown["gru"]),
                "misc": [0.0] * len(breakdown["misc"]),
            }
            self._v5_breakdown_acc = acc
        acc["n"] += 1
        for k in range(len(breakdown["gemm"])):
            acc["gemm"][k] += breakdown["gemm"][k]
            acc["score"][k] += breakdown["score"][k]
            acc["gru"][k] += breakdown["gru"][k]
            acc["misc"][k] += breakdown["misc"][k]
            if k < len(breakdown.get("score_main", [])):
                acc["score_main"][k] += breakdown["score_main"][k]
                acc["score_reduce"][k] += breakdown["score_reduce"][k]
        if acc["n"] % acc["print_every"] == 0:
            n = acc["n"]
            lines = ["[DFLASH v5 loop breakdown] per-step avg (ms) over %d calls:" % n]
            has_subscore = len(acc["score_main"]) > 0
            if has_subscore:
                header = "  k | proj_gemm | score(main+reduce) | gru_update | misc | step_total"
            else:
                header = "  k | proj_gemm | score_argmax | gru_update | misc | step_total"
            lines.append(header)
            total_gemm = 0.0
            total_score = 0.0
            total_gru = 0.0
            total_misc = 0.0
            total_score_main = 0.0
            total_score_reduce = 0.0
            for k in range(1, len(acc["gemm"])):
                g = acc["gemm"][k] / n
                s = acc["score"][k] / n
                r = acc["gru"][k] / n
                m = acc["misc"][k] / n
                t = g + s + r + m
                if has_subscore:
                    sm = acc["score_main"][k] / n
                    sr = acc["score_reduce"][k] / n
                    lines.append(
                        f"  {k:2d} | {g:9.4f} | {s:6.4f}({sm:5.4f}+{sr:5.4f}) | {r:10.4f} | {m:4.4f} | {t:10.4f}"
                    )
                    total_score_main += sm
                    total_score_reduce += sr
                else:
                    lines.append(
                        f"  {k:2d} | {g:9.4f} | {s:12.4f} | {r:10.4f} | {m:4.4f} | {t:10.4f}"
                    )
                total_gemm += g
                total_score += s
                total_gru += r
                total_misc += m
            if has_subscore:
                lines.append(
                    f"  14-step sum | proj_gemm={total_gemm:.4f} "
                    f"score={total_score:.4f}(main={total_score_main:.4f}+reduce={total_score_reduce:.4f}) "
                    f"gru={total_gru:.4f} misc={total_misc:.4f} "
                    f"total={total_gemm+total_score+total_gru+total_misc:.4f}"
                )
            else:
                lines.append(
                    f"  14-step sum | proj_gemm={total_gemm:.4f} score={total_score:.4f} "
                    f"gru={total_gru:.4f} misc={total_misc:.4f} total={total_gemm+total_score+total_gru+total_misc:.4f}"
                )
            logger.warning("\n".join(lines))

    def _get_or_capture_v5_loop_graph(
        self,
        *,
        bs: int,
        num_draft: int,
        emb_dim: int,
        gru_hidden: int,
        num_org: int,
        org_vocab_start: int,
        z_dtype: torch.dtype,
        logits_dtype: torch.dtype,
        device: torch.device,
        state: dict,
        gru_input_table: torch.Tensor,
        hidden_size: int,
        lm_head_weight: torch.Tensor,
        candidate_pool_size: int = 0,
    ):
        """Capture (or return cached) CUDA graph for the 13-iteration v5 rollout loop.

        Pool keyed by (bs, num_draft, dims, dtypes, fused-toggles) so each
        concurrency/dtype combination gets its own graph. Static buffers
        (z_proj_all, base_logits, gru_h_in, out) are filled with copy_() before
        replay; out is read after replay. The entry holds an `_static_refs`
        list pinning every tensor referenced by the captured graph so the
        caching allocator cannot recycle their bytes.

        Uses the current best v5 rollout path by default: Triton fused scoring,
        table-based GRU input, in-kernel GRU bias, CUDA graph replay, and an
        optional candidate pool for bias scoring.
        """
        pool = getattr(self, "_v5_loop_graph_pool", None)
        if pool is None:
            pool = {}
            self._v5_loop_graph_pool = pool
        # Keep the runtime path on the measured best configuration. The older
        # 0/1 experimental variants are no longer exposed as production knobs.
        use_fused = True
        use_table_fused = True
        use_gru_bias_fused = True
        use_precompute_mid = False
        use_full_graph = False
        candidate_pool_size = max(0, int(candidate_pool_size))
        use_candidate_pool = candidate_pool_size > 0
        disable_loop_graph = False
        breakdown_enabled = False
        ablate_no_gru = False
        ablate_no_score = False
        # CUDA graph baked-in pointers must outlive the entry; the cache key
        # therefore includes every shape/dtype/branch toggle that changes the
        # captured node set. (bs alone would alias different num_draft / dtype
        # configurations onto the same dangling-graph entry.)
        pool_key = (
            bs,
            num_draft,
            emb_dim,
            gru_hidden,
            num_org,
            org_vocab_start,
            z_dtype,
            logits_dtype,
            use_fused,
            use_table_fused,
            use_gru_bias_fused,
            use_precompute_mid,
            use_full_graph,
            use_candidate_pool,
            candidate_pool_size,
            hidden_size,
            disable_loop_graph,
        )
        entry = pool.get(pool_key)
        if entry is not None:
            return entry

        G = gru_hidden

        # Inputs (filled by copy_ before replay).
        z_proj_buf = torch.empty(
            (bs, num_draft, emb_dim), dtype=z_dtype, device=device
        )
        base_logits_buf = torch.empty(
            (bs, num_draft, num_org), dtype=logits_dtype, device=device
        )
        gru_h_buf = torch.empty((bs, G), dtype=z_dtype, device=device)
        out_buf = torch.empty((bs, num_draft), dtype=torch.long, device=device)

        if use_fused:
            block_v = int(os.environ.get("DFLASH_V5_BLOCK_V", "512"))
            block_m = int(os.environ.get("DFLASH_V5_BLOCK_M", "32"))
            score_num_warps = int(os.environ.get("DFLASH_V5_SCORE_NUM_WARPS", "4"))
            score_num_stages = int(os.environ.get("DFLASH_V5_SCORE_NUM_STAGES", "3"))
            candidate_block_c = int(os.environ.get("DFLASH_V5_CANDIDATE_BLOCK_C", "512"))
            num_v_blocks_shard = (num_org + block_v - 1) // block_v
            num_score_blocks = (
                (candidate_pool_size + candidate_block_c - 1) // candidate_block_c
                if use_candidate_pool else num_v_blocks_shard
            )
            # The measured best path inlines index_select into the GRU cell kernel.
            w_s_T = state["w_s"].T.contiguous()
            w_hh_T = state["w_hh"].T.contiguous()
            # Merge the two per-step GEMMs (h @ w_s_T -> s_proj, h @ w_hh_T -> gh)
            # into one wider matmul [G] x [G, emb_dim + 3G]. Same input, same
            # math; cuts a launch and lets cuBLAS pick a better tile.
            w_sh_T = torch.cat([w_s_T, w_hh_T], dim=1).contiguous()
            sh_buf = torch.empty(
                (bs, emb_dim + 3 * G), dtype=z_dtype, device=device
            )
            s_proj_buf = sh_buf[:, :emb_dim]
            gh_buf = sh_buf[:, emb_dim:]
            mid_buf = (
                torch.empty((bs, emb_dim), dtype=z_dtype, device=device)
                if use_precompute_mid else None
            )
            h_new_buf = torch.empty((bs, G), dtype=z_dtype, device=device)
            argmax_val_buf = torch.empty(
                (bs, num_score_blocks), dtype=torch.float32, device=device,
            )
            argmax_idx_buf = torch.empty(
                (bs, num_score_blocks), dtype=torch.int32, device=device,
            )
            candidate_ids_buf = (
                torch.empty((bs, candidate_pool_size), dtype=torch.long, device=device)
                if use_candidate_pool else None
            )
            if candidate_ids_buf is not None:
                candidate_ids_buf.zero_()
            local_tok_buf = torch.empty((bs,), dtype=torch.long, device=device)
            gi_buf = (
                None if use_table_fused
                else torch.empty((bs, 3 * G), dtype=z_dtype, device=device)
            )
            fc2_w_shard = state["fc2_weight"][
                org_vocab_start:org_vocab_start + num_org
            ].contiguous()
            fc2_b_shard = (
                state["fc2_bias"][
                    org_vocab_start:org_vocab_start + num_org
                ].contiguous()
                if state["fc2_bias"] is not None else None
            )
            b_hh_static = state["b_hh"]

            # Every tensor below is read by baked-in pointers inside the captured
            # graph. Without an explicit reference held by `entry`, the PyTorch
            # caching allocator is free to hand these CUDA bytes to other ops
            # once this function returns, which surfaces as illegal-memory-access
            # on the second replay. Keep this list complete.
            static_refs = [
                z_proj_buf, base_logits_buf, gru_h_buf, out_buf,
                sh_buf, h_new_buf,
                argmax_val_buf, argmax_idx_buf, local_tok_buf,
                fc2_w_shard, w_sh_T,
                gru_input_table,
            ]
            if mid_buf is not None:
                static_refs.append(mid_buf)
            if fc2_b_shard is not None:
                static_refs.append(fc2_b_shard)
            if b_hh_static is not None:
                static_refs.append(b_hh_static)
            if candidate_ids_buf is not None:
                static_refs.append(candidate_ids_buf)
            if gi_buf is not None:
                static_refs.append(gi_buf)

            def run_loop(out_target):
                h_state = gru_h_buf
                breakdown = None
                if breakdown_enabled:
                    breakdown = {
                        "gemm": [0.0] * num_draft,
                        "score": [0.0] * num_draft,
                        "score_main": [0.0] * num_draft,
                        "score_reduce": [0.0] * num_draft,
                        "gru": [0.0] * num_draft,
                        "misc": [0.0] * num_draft,
                    }
                    ev = [
                        [torch.cuda.Event(enable_timing=True) for _ in range(5)]
                        for _ in range(num_draft)
                    ]
                for k in range(1, num_draft):
                    if breakdown_enabled:
                        ev[k][0].record()
                    torch.matmul(h_state, w_sh_T, out=sh_buf)
                    if breakdown_enabled:
                        ev[k][1].record()
                    if not ablate_no_score:
                        score_ev = None
                        if breakdown_enabled:
                            score_ev = [None, None]
                        if use_candidate_pool:
                            assert candidate_ids_buf is not None
                            fused_silu_fc2_candidate_argmax(
                                z_proj=z_proj_buf[:, k, :],
                                s_proj=s_proj_buf,
                                fc2_weight=fc2_w_shard,
                                fc2_bias=fc2_b_shard,
                                base_logits=base_logits_buf[:, k, :],
                                candidate_ids=candidate_ids_buf,
                                out_val=argmax_val_buf,
                                out_idx=argmax_idx_buf,
                                final_token=local_tok_buf,
                                block_c=candidate_block_c,
                                block_m=block_m,
                                num_warps=score_num_warps,
                                num_stages=score_num_stages,
                                _profile_events=score_ev,
                            )
                        elif use_precompute_mid:
                            assert mid_buf is not None
                            compute_silu_sum(
                                z_proj=z_proj_buf[:, k, :],
                                s_proj=s_proj_buf,
                                mid_out=mid_buf,
                            )
                            fused_mid_fc2_argmax(
                                mid_proj=mid_buf,
                                fc2_weight=fc2_w_shard,
                                fc2_bias=fc2_b_shard,
                                base_logits=base_logits_buf[:, k, :],
                                out_val=argmax_val_buf,
                                out_idx=argmax_idx_buf,
                                final_token=local_tok_buf,
                                block_v=block_v,
                                block_m=block_m,
                                num_warps=score_num_warps,
                                num_stages=score_num_stages,
                                _profile_events=score_ev,
                            )
                        else:
                            fused_silu_fc2_argmax(
                                z_proj=z_proj_buf[:, k, :],
                                s_proj=s_proj_buf,
                                fc2_weight=fc2_w_shard,
                                fc2_bias=fc2_b_shard,
                                base_logits=base_logits_buf[:, k, :],
                                out_val=argmax_val_buf,
                                out_idx=argmax_idx_buf,
                                final_token=local_tok_buf,
                                block_v=block_v,
                                block_m=block_m,
                                num_warps=score_num_warps,
                                num_stages=score_num_stages,
                                _profile_events=score_ev,
                            )
                        if breakdown_enabled and score_ev is not None:
                            # score_ev[0] recorded after main, score_ev[1] after reduce
                            breakdown["score_main"][k] = ev[k][1].elapsed_time(
                                score_ev[0]
                            )
                            breakdown["score_reduce"][k] = score_ev[0].elapsed_time(
                                score_ev[1]
                            )
                    else:
                        local_tok_buf.fill_(0)
                    if breakdown_enabled:
                        ev[k][2].record()
                    tok_full = local_tok_buf + org_vocab_start
                    out_target[:, k] = tok_full
                    if breakdown_enabled:
                        ev[k][3].record()
                    if k + 1 < num_draft:
                        if not ablate_no_gru:
                            gh_bias = b_hh_static
                            if b_hh_static is not None and not use_gru_bias_fused:
                                gh_buf.add_(b_hh_static)
                                gh_bias = None
                            if use_table_fused:
                                fused_gru_cell_from_table(
                                    tok_full=tok_full,
                                    gru_input_table=gru_input_table,
                                    gh=gh_buf,
                                    gh_bias=gh_bias,
                                    h_state=h_state, h_out=h_new_buf,
                                )
                            else:
                                torch.index_select(
                                    gru_input_table, 0, tok_full, out=gi_buf
                                )
                                fused_gru_cell(
                                    gi=gi_buf, gh=gh_buf, gh_bias=gh_bias,
                                    h_state=h_state, h_out=h_new_buf,
                                )
                            h_state = h_new_buf
                    if breakdown_enabled:
                        ev[k][4].record()
                if breakdown_enabled:
                    torch.cuda.synchronize()
                    for k in range(1, num_draft):
                        breakdown["gemm"][k] = ev[k][0].elapsed_time(ev[k][1])
                        breakdown["score"][k] = ev[k][1].elapsed_time(ev[k][2])
                        breakdown["misc"][k] = ev[k][2].elapsed_time(ev[k][3])
                        breakdown["gru"][k] = ev[k][3].elapsed_time(ev[k][4])
                return breakdown
        else:
            # Pure cuBLAS+elementwise version (verified to work).
            # Optimization: replace per-step `embed_module(tok)` + `F.linear(W_ih)`
            # with a single `index_select(gru_input_table)`. The table is
            # `embed_weight @ W_ih.T + b_ih`, precomputed once per model load.
            w_s_static = state["w_s"]
            w_hh_static = state["w_hh"]
            b_hh_static = state["b_hh"]
            fc2_w_static = state["fc2_weight"]
            fc2_b_static = state["fc2_bias"]
            static_refs = [
                z_proj_buf, base_logits_buf, gru_h_buf, out_buf,
                w_s_static, w_hh_static, fc2_w_static, gru_input_table,
            ]
            if b_hh_static is not None:
                static_refs.append(b_hh_static)
            if fc2_b_static is not None:
                static_refs.append(fc2_b_static)

            def run_loop(out_target):
                gru_h_local = gru_h_buf.clone()
                for k in range(1, num_draft):
                    s_proj = torch.nn.functional.linear(
                        gru_h_local, w_s_static, None
                    )
                    mid = torch.nn.functional.silu(z_proj_buf[:, k, :] + s_proj)
                    bias_k = torch.nn.functional.linear(
                        mid, fc2_w_static, fc2_b_static
                    )
                    shard_end = org_vocab_start + num_org
                    bias_local = bias_k[:, org_vocab_start:shard_end]
                    logits_k = base_logits_buf[:, k, :] + bias_local.to(logits_dtype)
                    local_arg = torch.argmax(logits_k, dim=-1)
                    tok = (local_arg + org_vocab_start).to(torch.long)
                    out_target[:, k] = tok
                    if k + 1 < num_draft:
                        gi = torch.index_select(gru_input_table, 0, tok)
                        gh = torch.nn.functional.linear(
                            gru_h_local, w_hh_static, b_hh_static
                        )
                        r = torch.sigmoid(gi[:, :G] + gh[:, :G])
                        z_gate = torch.sigmoid(gi[:, G:2 * G] + gh[:, G:2 * G])
                        n = torch.tanh(gi[:, 2 * G:] + r * gh[:, 2 * G:])
                        gru_h_local = (1.0 - z_gate) * n + z_gate * gru_h_local

        # The full-graph variant was measured as not materially better than the
        # eager prologue plus captured rollout loop, so production stays on the
        # simpler graph boundary.
        if use_full_graph:
            z_buf = torch.empty(
                (bs, num_draft, hidden_size), dtype=z_dtype, device=device
            )
            verified_id_buf = torch.empty((bs,), dtype=torch.long, device=device)
            slot_1_local_buf = torch.empty((bs,), dtype=torch.long, device=device)
            slot_1_buf = torch.empty((bs,), dtype=torch.long, device=device)
            h_init_a = torch.empty((bs, G), dtype=z_dtype, device=device)
            h_init_b = torch.empty((bs, G), dtype=z_dtype, device=device)
            gh_init_buf = torch.empty((bs, 3 * G), dtype=z_dtype, device=device)

            lm_head_T = lm_head_weight[:num_org].T.contiguous()
            w_z_T = state["w_z"].T.contiguous()
            w_hh_T_init = state["w_hh"].T.contiguous()
            b1_static = state["b1"]

            full_static_refs = list(static_refs) + [
                z_buf, verified_id_buf,
                slot_1_local_buf, slot_1_buf,
                h_init_a, h_init_b, gh_init_buf,
                lm_head_T, w_z_T, w_hh_T_init,
            ]
            if b1_static is not None:
                full_static_refs.append(b1_static)

            def run_full(out_target):
                # 1) Dense base logits over the org-vocab shard.
                torch.matmul(z_buf, lm_head_T, out=base_logits_buf)
                # 2) slot_1 = argmax(base_logits[:, 0]) + org_vocab_start.
                torch.argmax(base_logits_buf[:, 0, :], dim=-1, out=slot_1_local_buf)
                torch.add(slot_1_local_buf, org_vocab_start, out=slot_1_buf)
                out_target[:, 0] = slot_1_buf
                # 3) Manual 2-step GRU init via gru_input_table lookups.
                gi_v = torch.index_select(gru_input_table, 0, verified_id_buf)
                gi_s = torch.index_select(gru_input_table, 0, slot_1_buf)
                h_init_a.zero_()
                h_in = h_init_a
                h_out = h_init_b
                for gi_t in (gi_v, gi_s):
                    torch.matmul(h_in, w_hh_T_init, out=gh_init_buf)
                    if b_hh_static is not None:
                        gh_init_buf.add_(b_hh_static)
                    gi_r = gi_t[:, :G]
                    gi_z = gi_t[:, G:2 * G]
                    gi_n = gi_t[:, 2 * G:]
                    gh_r = gh_init_buf[:, :G]
                    gh_z = gh_init_buf[:, G:2 * G]
                    gh_n = gh_init_buf[:, 2 * G:]
                    r = torch.sigmoid(gi_r + gh_r)
                    z_gate = torch.sigmoid(gi_z + gh_z)
                    n = torch.tanh(gi_n + r * gh_n)
                    h_new = (1.0 - z_gate) * n + z_gate * h_in
                    h_out.copy_(h_new)
                    h_in, h_out = h_out, h_in
                gru_h_buf.copy_(h_in)
                # 4) z_proj_all = z @ w_z.T (+ b1).
                torch.matmul(z_buf, w_z_T, out=z_proj_buf)
                if b1_static is not None:
                    z_proj_buf.add_(b1_static)
                # 5) The original 14-step loop.
                run_loop(out_target)
        else:
            full_static_refs = static_refs
            run_full = run_loop  # kept for symmetry; outer code branches on use_full_graph

        if disable_loop_graph:
            entry = {
                "z_proj_buf": z_proj_buf,
                "base_logits_buf": base_logits_buf,
                "gru_h_buf": gru_h_buf,
                "out_buf": out_buf,
                "run_full": run_full,
                "_static_refs": full_static_refs,
            }
            if use_full_graph:
                entry["z_buf"] = z_buf
                entry["verified_id_buf"] = verified_id_buf
            if use_candidate_pool:
                entry["candidate_ids_buf"] = candidate_ids_buf
            pool[pool_key] = entry
            logger.warning(
                "[DFLASH v5] eager loop path (no graph) for bs=%d num_draft=%d",
                bs, num_draft,
            )
            return entry

        # Warmup outside graph.
        warmup_out = torch.empty((bs, num_draft), dtype=torch.long, device=device)
        for _ in range(2):
            run_full(warmup_out)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_full(out_buf)

        entry = {
            "graph": graph,
            "z_proj_buf": z_proj_buf,
            "base_logits_buf": base_logits_buf,
            "gru_h_buf": gru_h_buf,
            "out_buf": out_buf,
            # Holds Python references to every tensor with a baked-in pointer
            # inside the captured graph. Without this, locals here go out of
            # scope and the caching allocator may reuse their CUDA bytes,
            # causing illegal memory access on graph replay.
            "_static_refs": full_static_refs,
        }
        if use_full_graph:
            entry["z_buf"] = z_buf
            entry["verified_id_buf"] = verified_id_buf
        if use_candidate_pool:
            entry["candidate_ids_buf"] = candidate_ids_buf
        pool[pool_key] = entry
        logger.warning(
            "[DFLASH v5] captured CUDA graph (%s) for v5 rollout loop bs=%d num_draft=%d",
            (
                "Triton-candidate-pool"
                if use_candidate_pool else ("Triton-fused" if use_fused else "cuBLAS")
            ),
            bs, num_draft,
        )
        return entry

    def _v5_rollout_draft_block(
        self,
        *,
        draft_hidden: torch.Tensor,
        verified_id: torch.Tensor,
        target_model,
        lm_head,
    ) -> torch.Tensor:
        """Sequential v5 rollout to produce `block_size - 1` draft tokens.

        Algorithm mirrors dflash benchmark.py:244-264 for both
        shift_label=True and shift_label=False. SGLang's draft block reserves
        slot 0 for the current verified token and emits slots 1..block_size-1:
          1. Select z so z[:, 0] predicts slot_1 under the checkpoint's label
             alignment.
          2. slot_1 = argmax(base_logits[0]).
          3. Initialize prefix_gru on embeddings of [verified_id, slot_1] to get gru_h.
          4. For k = 1..block_size-2:
                 bias = embed_proj(cat(z_k, gru_h))
                 slot_{k+1} = argmax(base_logits[k] + bias)
                 gru_h = prefix_gru_step(embed(slot_{k+1}))

        Implementation note: the per-step embed_proj/GRU calls have non-trivial
        Python + cuDNN launch overhead.  We pre-split fc1.weight along its input
        axis so z @ W_z (the part that doesn't depend on the GRU state) is
        batched outside the loop, and we replace nn.GRU(seq_len=1) with a manual
        GRU cell to skip cuDNN setup.

        Args:
            draft_hidden: [B, block_size, hidden_size] draft model output (post-norm).
            verified_id: [B] current verified token per request (int64).
            target_model: the SGLang target model (for embed_tokens).
            lm_head: vocab-parallel lm_head (used to compute dense base logits).

        Returns:
            [B, block_size - 1] int64 tensor of sampled draft tokens.
        """
        bs, total_slots, hidden_size = draft_hidden.shape
        if total_slots != self.block_size:
            raise RuntimeError(
                f"DFLASH v5 expected draft_hidden block dim={self.block_size}, "
                f"got {total_slots}."
            )
        num_draft = self.block_size - 1  # 15 for block_size=16
        if num_draft <= 0:
            raise RuntimeError(
                f"DFLASH v5 requires block_size > 1, got {self.block_size}."
            )

        device = draft_hidden.device
        tp_size = int(get_tp_group().world_size)

        weight = lm_head.weight
        shard = lm_head.shard_indices
        num_added = int(shard.num_added_elements)
        if num_added != 0:
            raise NotImplementedError(
                "DFLASH Domino rollout does not yet handle added-vocab lm_head shards."
            )
        org_vocab_start = int(shard.org_vocab_start_index)
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        if num_org <= 0:
            raise RuntimeError("DFLASH lm_head has empty base vocab shard.")

        candidate_pool_size = max(
            0, int(os.environ.get("DFLASH_V5_CANDIDATE_POOL", "1024"))
        )
        if candidate_pool_size > 0 and candidate_pool_size >= num_org:
            if not getattr(self, "_v5_candidate_full_vocab_warned", False):
                logger.warning(
                    "DFLASH_V5_CANDIDATE_POOL=%d is >= local vocab size %d; "
                    "using full-vocab v5 scoring instead.",
                    candidate_pool_size, num_org,
                )
                self._v5_candidate_full_vocab_warned = True
            candidate_pool_size = 0

        draft_model = self.draft_model
        embed_module = target_model.get_input_embeddings()

        prof_enabled = os.environ.get("DFLASH_PROF_V5") == "1"
        if prof_enabled:
            prof = getattr(self, "_v5_prof", None)
            if prof is None:
                window_size = int(os.environ.get("DFLASH_PROF_V5_WINDOW", "100"))
                prof = {
                    "n": 0,
                    "base_logits_ms": 0.0,
                    "init_gru_ms": 0.0,
                    "loop_ms": 0.0,
                    "total_ms": 0.0,
                    "print_every": 50,
                    "window_size": window_size,
                    "win_base_logits": [],
                    "win_init_gru": [],
                    "win_loop": [],
                    "win_total": [],
                    "evt_total_start": torch.cuda.Event(enable_timing=True),
                    "evt_total_end": torch.cuda.Event(enable_timing=True),
                    "evt_base_start": torch.cuda.Event(enable_timing=True),
                    "evt_base_end": torch.cuda.Event(enable_timing=True),
                    "evt_init_start": torch.cuda.Event(enable_timing=True),
                    "evt_init_end": torch.cuda.Event(enable_timing=True),
                    "evt_loop_start": torch.cuda.Event(enable_timing=True),
                    "evt_loop_end": torch.cuda.Event(enable_timing=True),
                    "last_call_ts": None,
                    "warmup_reset_done": False,
                }
                self._v5_prof = prof
            # Optional one-shot reset after the first idle gap > 1s. The bench
            # script's /flush_cache + warmup loop creates exactly such a gap
            # before the timed phase starts; this lets prof totals reflect
            # only the timed-phase rollouts. Off by default.
            reset_after_warmup = os.environ.get(
                "DFLASH_PROF_V5_RESET_AFTER_WARMUP"
            ) == "1"
            now_ts = time.perf_counter()
            if (
                reset_after_warmup
                and not prof["warmup_reset_done"]
                and prof["last_call_ts"] is not None
                and now_ts - prof["last_call_ts"] > 1.0
                and prof["n"] > 0
            ):
                prof["n"] = 0
                prof["base_logits_ms"] = 0.0
                prof["init_gru_ms"] = 0.0
                prof["loop_ms"] = 0.0
                prof["total_ms"] = 0.0
                prof["win_base_logits"] = []
                prof["win_init_gru"] = []
                prof["win_loop"] = []
                prof["win_total"] = []
                prof["warmup_reset_done"] = True
                logger.warning(
                    "[DFLASH v5 prof] reset counters after %.2fs idle gap (warmup boundary)",
                    now_ts - prof["last_call_ts"],
                )
            prof["last_call_ts"] = now_ts
            prof["evt_total_start"].record()
            prof["evt_base_start"].record()

        # shift_label=True: draft_hidden[:, i, :] predicts token at position i+1.
        # shift_label=False: draft_hidden[:, i, :] predicts token at position i.
        # For the draft block, position 0 is verified_id; positions 1..block_size-1
        # are the draft slots.  With shift_label we need draft_hidden[:, 0, :]
        # to predict slot_1, so we slice [:num_draft]; otherwise we slice [1:].
        if getattr(draft_model, "shift_label", False):
            z = draft_hidden[:, :num_draft, :].contiguous()  # [B, num_draft, hidden]
        else:
            z = draft_hidden[:, 1:, :].contiguous()  # [B, num_draft, hidden]

        state = draft_model.get_v5_rollout_state()
        if tp_size != 1:
            return self._v5_rollout_draft_block_tp_eager(
                z=z,
                verified_id=verified_id,
                target_model=target_model,
                lm_head=lm_head,
                org_vocab_start=org_vocab_start,
                num_org=num_org,
                num_org_padded=num_org_padded,
                state=state,
            )

        G = state["gru_hidden_size"]
        emb_dim = int(state["w_z"].shape[0])
        z_for_dtype = z.to(weight.dtype) if z.dtype != weight.dtype else z
        gru_input_table = draft_model.get_v5_gru_input_proj_table(
            embed_module.weight
        )

        use_full_graph = False

        if not use_full_graph:
            # Original eager-prologue path.
            z_flat = z_for_dtype.reshape(bs * num_draft, hidden_size)
            base_logits = torch.matmul(z_flat, weight[:num_org].T).view(
                bs, num_draft, num_org
            )
            if prof_enabled:
                prof["evt_base_end"].record()
                prof["evt_init_start"].record()

            candidate_ids = None
            if candidate_pool_size > 0:
                pool_source = (
                    base_logits[:, 1:, :]
                    if int(base_logits.shape[1]) > 1 else base_logits[:, :1, :]
                )
                pool_logits = pool_source.max(dim=1).values
                candidate_ids = torch.topk(
                    pool_logits, k=candidate_pool_size, dim=-1
                ).indices.contiguous()

            slot_local_arg = torch.argmax(base_logits[:, 0, :], dim=-1)
            slot_1 = (slot_local_arg + org_vocab_start).to(torch.long)

            prefix_ids = torch.stack([verified_id.to(torch.long), slot_1], dim=1)

            use_fast_init = True
            check_init = False

            if use_fast_init or check_init:
                # Manual 2-step unroll using precomputed gru_input_table.
                w_hh = state["w_hh"]  # [3*G, G]
                b_hh = state["b_hh"]  # [3*G] or None
                w_hh_T = w_hh.T.to(dtype=z.dtype, device=device)
                h = torch.zeros((bs, G), dtype=z.dtype, device=device)
                for t_idx in range(prefix_ids.shape[1]):
                    gi = torch.index_select(gru_input_table, 0, prefix_ids[:, t_idx])
                    gh = torch.matmul(h, w_hh_T)
                    if b_hh is not None:
                        gh.add_(b_hh.to(z.dtype))
                    r = torch.sigmoid(gi[:, :G] + gh[:, :G])
                    z_gate = torch.sigmoid(gi[:, G : 2 * G] + gh[:, G : 2 * G])
                    n = torch.tanh(gi[:, 2 * G :] + r * gh[:, 2 * G :])
                    h = (1.0 - z_gate) * n + z_gate * h
                gru_h_fast = h

            if check_init:
                prefix_embeds = embed_module(prefix_ids).to(z.dtype)
                gru_h_ref = draft_model.v5_init_gru_hidden(prefix_embeds)
                diff = (gru_h_ref.float() - gru_h_fast.float()).abs()
                max_diff = float(diff.max())
                logger.warning(
                    "[DFLASH v5 fast_init check] hidden max_abs_diff=%.4e", max_diff
                )
                gru_h = gru_h_ref
            elif use_fast_init:
                gru_h = gru_h_fast
            else:
                prefix_embeds = embed_module(prefix_ids).to(z.dtype)
                gru_h = draft_model.v5_init_gru_hidden(prefix_embeds)

            z_proj_all = torch.nn.functional.linear(z, state["w_z"], state["b1"])
            if prof_enabled:
                prof["evt_init_end"].record()
                prof["evt_loop_start"].record()

            graph_entry = self._get_or_capture_v5_loop_graph(
                bs=bs,
                num_draft=num_draft,
                emb_dim=emb_dim,
                gru_hidden=G,
                num_org=num_org,
                org_vocab_start=org_vocab_start,
                z_dtype=z.dtype,
                logits_dtype=base_logits.dtype,
                device=device,
                state=state,
                gru_input_table=gru_input_table,
                hidden_size=hidden_size,
                lm_head_weight=weight,
                candidate_pool_size=candidate_pool_size,
            )
            graph_entry["z_proj_buf"].copy_(z_proj_all)
            graph_entry["base_logits_buf"].copy_(base_logits)
            if candidate_ids is not None:
                graph_entry["candidate_ids_buf"].copy_(candidate_ids)
            graph_entry["gru_h_buf"].copy_(gru_h)
            graph_entry["out_buf"][:, 0].copy_(slot_1)
            if graph_entry.get("graph") is not None:
                graph_entry["graph"].replay()
                out = graph_entry["out_buf"]
            else:
                breakdown = graph_entry["run_full"](graph_entry["out_buf"])
                out = graph_entry["out_buf"]
                if breakdown is not None:
                    self._accumulate_and_print_breakdown(breakdown)

            if check_init and graph_entry.get("graph") is not None:
                out_ref = out.clone()
                graph_entry["gru_h_buf"].copy_(gru_h_fast)
                graph_entry["graph"].replay()
                out_fast = graph_entry["out_buf"].clone()
                tok_diff = int((out_ref != out_fast).sum().item())
                if tok_diff > 0:
                    first_diff = int(
                        (out_ref[0] != out_fast[0]).nonzero(as_tuple=True)[0][0].item()
                    )
                    logger.warning(
                        "[DFLASH v5 fast_init check] draft tokens DIFFER: %d mismatches, first at k=%d",
                        tok_diff,
                        first_diff,
                    )
                else:
                    logger.warning(
                        "[DFLASH v5 fast_init check] draft tokens match exactly"
                    )
                out = out_ref
        else:
            # Full-graph path: base_logits, slot_1, GRU init, z_proj all run
            # inside the captured graph. We only feed z and verified_id.
            if prof_enabled:
                prof["evt_base_end"].record()
                prof["evt_init_start"].record()
                prof["evt_init_end"].record()
                prof["evt_loop_start"].record()
            graph_entry = self._get_or_capture_v5_loop_graph(
                bs=bs,
                num_draft=num_draft,
                emb_dim=emb_dim,
                gru_hidden=G,
                num_org=num_org,
                org_vocab_start=org_vocab_start,
                z_dtype=z.dtype,
                logits_dtype=weight.dtype,
                device=device,
                state=state,
                gru_input_table=gru_input_table,
                hidden_size=hidden_size,
                lm_head_weight=weight,
                candidate_pool_size=0,
            )
            graph_entry["z_buf"].copy_(z_for_dtype)
            graph_entry["verified_id_buf"].copy_(verified_id.to(torch.long))
            if graph_entry.get("graph") is not None:
                graph_entry["graph"].replay()
                out = graph_entry["out_buf"]
            else:
                breakdown = graph_entry["run_full"](graph_entry["out_buf"])
                out = graph_entry["out_buf"]
                if breakdown is not None:
                    self._accumulate_and_print_breakdown(breakdown)

        if prof_enabled:
            prof["evt_loop_end"].record()
            prof["evt_total_end"].record()
            torch.cuda.synchronize()
            base_logits_t = prof["evt_base_start"].elapsed_time(prof["evt_base_end"])
            init_gru_t = prof["evt_init_start"].elapsed_time(prof["evt_init_end"])
            loop_t = prof["evt_loop_start"].elapsed_time(prof["evt_loop_end"])
            total_t = prof["evt_total_start"].elapsed_time(prof["evt_total_end"])

            prof["base_logits_ms"] += base_logits_t
            prof["init_gru_ms"] += init_gru_t
            prof["loop_ms"] += loop_t
            prof["total_ms"] += total_t
            prof["n"] += 1

            ws = prof["window_size"]
            prof["win_base_logits"].append(base_logits_t)
            prof["win_init_gru"].append(init_gru_t)
            prof["win_loop"].append(loop_t)
            prof["win_total"].append(total_t)
            if len(prof["win_total"]) > ws:
                prof["win_base_logits"].pop(0)
                prof["win_init_gru"].pop(0)
                prof["win_loop"].pop(0)
                prof["win_total"].pop(0)

            if prof["n"] % prof["print_every"] == 0:
                n = prof["n"]
                win_n = len(prof["win_total"])
                win_total = sum(prof["win_total"]) / win_n
                win_base = sum(prof["win_base_logits"]) / win_n
                win_init = sum(prof["win_init_gru"]) / win_n
                win_loop = sum(prof["win_loop"]) / win_n
                logger.warning(
                    "[DFLASH v5 prof] bs=%d n_calls=%d "
                    "total=%.3fms (win %.3fms) base_logits=%.3fms (win %.3fms) "
                    "init_gru=%.3fms (win %.3fms) loop(14step)=%.3fms (win %.3fms)",
                    bs, n,
                    prof["total_ms"] / n, win_total,
                    prof["base_logits_ms"] / n, win_base,
                    prof["init_gru_ms"] / n, win_init,
                    prof["loop_ms"] / n, win_loop,
                )

        return out

    def _append_target_hidden_to_draft_kv(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInput,
    ) -> None:
        """Materialize the target hidden-state features into the draft KV cache.

        This must be run before exposing new tokens to radix cache (prefix hits), otherwise
        another request could reuse target KV indices without having draft KV values.
        """

        bs = batch.batch_size()
        device = self.model_runner.device

        if draft_input.target_hidden is None:
            raise RuntimeError(
                "DFLASH draft state missing target_hidden context features."
            )
        if draft_input.ctx_lens.numel() != bs:
            raise RuntimeError(
                f"DFLASH ctx_lens length mismatch: got {draft_input.ctx_lens.numel()} for bs={bs}."
            )
        if draft_input.draft_seq_lens.numel() != bs:
            raise RuntimeError(
                f"DFLASH draft_seq_lens length mismatch: got {draft_input.draft_seq_lens.numel()} for bs={bs}."
            )

        total_ctx = int(draft_input.target_hidden.shape[0])
        if total_ctx <= 0:
            draft_input.ctx_lens = torch.zeros_like(draft_input.ctx_lens)
            draft_input.target_hidden = draft_input.target_hidden[:0]
            return

        target_req_to_token = batch.req_to_token_pool.req_to_token
        draft_req_to_token = self.draft_model_runner.req_to_token_pool.req_to_token

        req_pool_indices = batch.req_pool_indices
        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)

        ctx_lens = draft_input.ctx_lens
        if ctx_lens.dtype != torch.int32:
            ctx_lens = ctx_lens.to(torch.int32)
        if ctx_lens.device != device:
            ctx_lens = ctx_lens.to(device, non_blocking=True)
        ctx_start = batch.seq_lens.to(torch.int64) - ctx_lens.to(torch.int64)

        if bs == 1:
            # Fast path for single request.
            max_ctx = int(total_ctx)
            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            pos2d = ctx_start[:, None] + r[None, :]  # [1, ctx]
            cache2d = target_req_to_token[req_pool_indices[:, None], pos2d]  # [1, ctx]
            ctx_cache_loc = cache2d.reshape(-1).to(torch.int64)  # [ctx]
            ctx_positions = pos2d.reshape(-1)  # [ctx]
        else:
            # In decode mode, ctx_lens <= block_size so we can skip the .item() sync.
            if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
                max_ctx = int(ctx_lens.max().item())
            else:
                max_ctx = int(self.block_size)
            if max_ctx <= 0:
                raise RuntimeError(f"DFLASH invalid max_ctx={max_ctx} for KV append.")

            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            r = r[None, :]  # [1, max_ctx]
            pos2d = ctx_start[:, None] + r  # [bs, max_ctx]
            mask = r < ctx_lens[:, None]

            # Batched gather of cache locations and positions.
            ctx_cache_loc = self._gather_req_to_token_masked(
                req_to_token=target_req_to_token,
                req_pool_indices=req_pool_indices,
                pos2d=pos2d,
                mask=mask,
                context="DFLASH target hidden KV append",
            )  # [sum(ctx_lens)]
            ctx_positions = pos2d[mask]  # [sum(ctx_lens)]

        with torch.inference_mode():
            ctx_hidden = self.draft_model.project_target_hidden(
                draft_input.target_hidden
            )  # [sum(ctx), hidden]
            if ctx_hidden.shape[0] != ctx_cache_loc.numel():
                raise RuntimeError(
                    f"DFLASH ctx_hidden/cache_loc mismatch: {ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}."
                )

            if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                try:
                    self._append_target_hidden_fused(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )
                except Exception as e:
                    logger.warning(
                        "DFLASH fused KV append failed; falling back to sequential path: %s",
                        e,
                    )
                    self._use_fused_kv_materialize = False
                    self._fused_kv_helper = None
                    self._append_target_hidden_sequential(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )
            else:
                self._append_target_hidden_sequential(
                    ctx_hidden, ctx_positions, ctx_cache_loc
                )

        if self.use_compact_draft_cache:
            new_draft_seq_lens = self._compute_compact_draft_seq_lens(batch.seq_lens)
            suffix_start = batch.seq_lens.to(torch.int64) - new_draft_seq_lens.to(
                torch.int64
            )
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=target_req_to_token,
                req_pool_indices=req_pool_indices,
                start=suffix_start,
                lengths=new_draft_seq_lens,
            )
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                draft_req_to_token,
                torch.zeros_like(new_draft_seq_lens),
                new_draft_seq_lens,
                suffix_cache_loc,
                bs,
            )
            draft_input.draft_seq_lens = new_draft_seq_lens
        else:
            # The draft KV cache must reflect the exact number of committed tokens.
            # In extend mode this is the prompt length; in decode mode it is the
            # current target seq_len (all committed tokens including the latest
            # bonus/verified token).  The draft model will recompute the verified
            # token as the first query of the next block, but the cache length
            # itself must stay consistent with req_to_token so attention sees the
            # correct prefix.
            draft_input.draft_seq_lens = batch.seq_lens.to(dtype=torch.int32)
        draft_input.ctx_lens = torch.zeros_like(ctx_lens)
        draft_input.target_hidden = draft_input.target_hidden[:0]

    def _append_target_hidden_sequential(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        for layer in self.draft_model.layers:
            attn = layer.self_attn
            k, v = attn.kv_proj_only(ctx_hidden)
            k = attn.apply_k_norm(k)
            k = attn.apply_k_rope(ctx_positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            self.draft_model_runner.token_to_kv_pool.set_kv_buffer(
                attn.attn,
                ctx_cache_loc,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
            )

    def _append_target_hidden_fused(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        """Fused KV materialization using batched projection + Triton kernel."""
        token_to_kv_pool = self.draft_model_runner.token_to_kv_pool
        layers = self.draft_model.layers

        def _write_layer_kv(
            layer_idx: int, cache_k: torch.Tensor, cache_v: torch.Tensor
        ) -> None:
            attn = layers[layer_idx].self_attn.attn
            token_to_kv_pool.set_kv_buffer(
                attn,
                ctx_cache_loc,
                cache_k,
                cache_v,
                attn.k_scale,
                attn.v_scale,
            )

        self._fused_kv_helper.materialize(
            ctx_hidden=ctx_hidden,
            positions=ctx_positions,
            write_layer_kv=_write_layer_kv,
        )

    def _update_target_mamba_state_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit Mamba intermediate states for accepted verify steps.

        During TARGET_VERIFY, Mamba kernels run with `disable_state_update=True` and
        cache per-step intermediate states. After acceptance, we need to commit the
        state corresponding to each request's last accepted step.
        """
        attn_backend = self.target_worker.model_runner.attn_backend
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            return

        accepted_steps = commit_lens.to(torch.int64) - 1
        mamba_steps_to_track = None

        if batch.mamba_track_indices is not None:
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            can_track_mask = to_track_mask & (
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
            mamba_steps_to_track = torch.where(
                can_track_mask,
                to_track_ith.to(torch.int64),
                torch.full_like(to_track_ith, -1, dtype=torch.int64),
            )

        attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise RuntimeError(
                "Invariant broken: DFLASH batch requested return_logprob, but scheduler should have rejected this request."
            )

        if isinstance(batch, ModelWorkerBatch):
            # Should not happen for spec-v1 (non-overlap) scheduling, but keep a sane fallback.
            return self.target_worker.forward_batch_generation(batch, **kwargs)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, **kwargs
            )
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DFLASH requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if (
                model_worker_batch.extend_seq_lens is None
                or model_worker_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "DFLASH expected extend_seq_lens / extend_prefix_lens to be populated in extend mode, but got None."
                )

            # Materialize the prompt tokens into the draft KV cache immediately. This is required
            # for radix cache support, since the scheduler may update radix after prefill returns.
            device = next_token_ids.device

            def _to_int32_device_tensor(x, *, device=device):
                if isinstance(x, torch.Tensor):
                    if x.device != device:
                        x = x.to(device, non_blocking=True)
                    return x if x.dtype == torch.int32 else x.to(torch.int32)
                return torch.tensor(x, dtype=torch.int32, device=device)

            extend_seq_lens = _to_int32_device_tensor(
                model_worker_batch.extend_seq_lens
            )
            draft_input = DFlashDraftInput(
                verified_id=next_token_ids.to(torch.int64),
                target_hidden=logits_output.hidden_states,
                ctx_lens=extend_seq_lens,
                draft_seq_lens=(
                    torch.zeros_like(extend_seq_lens)
                    if self.use_compact_draft_cache
                    else _to_int32_device_tensor(model_worker_batch.extend_prefix_lens)
                ),
            )
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            batch.spec_info = draft_input

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_drafts=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        # Decode / target-verify stage.
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInput):
            raise RuntimeError(
                "DFLASH decode requires DFlashDraftInput state on the running batch. "
                "This usually means the request did not complete the prefill stage."
            )

        self._prepare_for_speculative_decoding(batch, draft_input)

        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        assert isinstance(verify_input, DFlashVerifyInput)
        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )

        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, **kwargs
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            num_accepted_drafts_per_req_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
        )
        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )

        # Update draft state for the next iteration. Also materialize the committed verify tokens
        # into the draft KV cache immediately so radix cache entries are safe to reuse.
        draft_input.verified_id = new_verified_id
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens = commit_lens
        self._append_target_hidden_to_draft_kv(batch, draft_input)
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        num_accepted_drafts = sum(num_accepted_drafts_per_req_cpu)
        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "DFLASH verify completed. num_accepted_drafts_per_req=%s",
                num_accepted_drafts_per_req_cpu,
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=new_verified_id,
            num_accepted_drafts=num_accepted_drafts,
            num_accepted_drafts_per_req_cpu=num_accepted_drafts_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
