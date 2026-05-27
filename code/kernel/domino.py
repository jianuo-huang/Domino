import argparse
import statistics
from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def unwrap_domino_mlp(
    embed_proj: nn.Module,
) -> Tuple[nn.Linear, nn.Module, nn.Linear]:
    """
    Domino embed_proj should be:

        nn.Sequential(
            nn.Linear(hidden_size + gru_hidden_dim, mlp_hidden_dim),
            ... middle ...,
            nn.Linear(mlp_hidden_dim, vocab_size)
        )

    Input is concat([z_i, s_i]).
    """

    if not isinstance(embed_proj, nn.Sequential):
        raise TypeError(
            "DominoGraphRunner requires draft_model.embed_proj to be nn.Sequential."
        )

    modules = list(embed_proj.children())

    if len(modules) < 2:
        raise ValueError("embed_proj must contain at least two layers.")

    if not isinstance(modules[0], nn.Linear):
        raise ValueError("embed_proj first layer must be nn.Linear.")

    if not isinstance(modules[-1], nn.Linear):
        raise ValueError("embed_proj last layer must be nn.Linear.")

    fc1 = modules[0]
    middle = nn.Sequential(*modules[1:-1]) if len(modules) > 2 else nn.Identity()
    fc2 = modules[-1]

    return fc1, middle, fc2


# -----------------------------
# Triton kernels (inline, no external deps)
# -----------------------------

if HAS_TRITON:
    @triton.jit
    def _v5_fused_gemv_argmax_kernel(
        z_ptr, s_ptr,
        fc2_ptr, base_ptr, fc2_bias_ptr, gate_ptr,
        out_val_ptr, out_idx_ptr,
        M, V,
        stride_z_b, stride_z_m,
        stride_s_b, stride_s_m,
        stride_f2_v, stride_f2_m,
        stride_base_b, stride_base_v,
        stride_out_b, stride_out_v,
        BLOCK_V: tl.constexpr,
        BLOCK_M: tl.constexpr,
        HAS_FC2_BIAS: tl.constexpr,
        HAS_GATE: tl.constexpr,
    ):
        """
        Fused: SiLU(z + s) @ fc2.T + [fc2_bias] + base → [gate *] → per-block argmax.
        Each block handles BLOCK_V vocab entries.
        """
        pid_v = tl.program_id(0)
        pid_b = tl.program_id(1)

        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V

        acc = tl.zeros([BLOCK_V], dtype=tl.float32)

        for m_start in range(0, M, BLOCK_M):
            offs_m = m_start + tl.arange(0, BLOCK_M)
            mask_m = offs_m < M

            # SiLU(z + s) on the fly
            z_ptrs = z_ptr + pid_b * stride_z_b + offs_m * stride_z_m
            s_ptrs = s_ptr + pid_b * stride_s_b + offs_m * stride_s_m
            z_block = tl.load(z_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            s_block = tl.load(s_ptrs, mask=mask_m, other=0.0).to(tl.float32)

            preact = z_block + s_block
            mid_block = preact * tl.sigmoid(preact)

            # GEMV tile
            fc2_ptrs = (fc2_ptr
                        + offs_v[:, None] * stride_f2_v
                        + offs_m[None, :] * stride_f2_m)
            fc2_block = tl.load(
                fc2_ptrs,
                mask=mask_v[:, None] & mask_m[None, :],
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)

            acc += tl.sum(fc2_block * mid_block[None, :], axis=1)

        # Add fc2_bias if present
        if HAS_FC2_BIAS:
            bias_ptrs = fc2_bias_ptr + offs_v
            bias_block = tl.load(bias_ptrs, mask=mask_v, other=0.0).to(tl.float32)
            acc += bias_block

        # Add base_logits
        base_ptrs = base_ptr + pid_b * stride_base_b + offs_v * stride_base_v
        base_block = tl.load(
            base_ptrs,
            mask=mask_v,
            other=-float("inf"),
            cache_modifier=".cg",
        ).to(tl.float32)

        acc += base_block

        # Apply gate if present (scalar per batch, broadcast to V)
        if HAS_GATE:
            gate_val = tl.load(gate_ptr + pid_b).to(tl.float32)
            acc = gate_val * acc

        acc = tl.where(mask_v, acc, -float("inf"))

        local_max_val = tl.max(acc, axis=0)
        local_max_idx = tl.argmax(acc, axis=0)
        global_max_idx = pid_v * BLOCK_V + local_max_idx

        out_offs = pid_b * stride_out_b + pid_v * stride_out_v
        tl.store(out_val_ptr + out_offs, local_max_val)
        tl.store(out_idx_ptr + out_offs, global_max_idx)

    @triton.jit
    def _v5_reduce_argmax_kernel(
        out_val_ptr, out_idx_ptr, final_token_ptr,
        N,
        stride_val_b, stride_val_n,
        stride_idx_b, stride_idx_n,
        stride_final_b,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        best_val = -float("inf")
        best_idx = 0

        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            val_ptrs = out_val_ptr + pid_b * stride_val_b + offs_n * stride_val_n
            vals = tl.load(val_ptrs, mask=mask_n, other=-float("inf"))

            local_pos = tl.argmax(vals, axis=0)
            local_val = tl.max(vals, axis=0)

            local_idx = tl.load(
                out_idx_ptr + pid_b * stride_idx_b
                + (start_n + local_pos) * stride_idx_n
            )

            take = local_val > best_val
            best_val = tl.where(take, local_val, best_val)
            best_idx = tl.where(take, local_idx, best_idx)

        tl.store(final_token_ptr + pid_b * stride_final_b, best_idx)

    @triton.jit
    def _v5_fused_gru_cell_kernel(
        gi_ptr, gh_ptr, h_ptr, h_out_ptr,
        B, G,
        stride_gi_b, stride_gi_g,
        stride_gh_b, stride_gh_g,
        stride_h_b, stride_h_g,
        stride_hout_b, stride_hout_g,
        BLOCK_G: tl.constexpr,
    ):
        """
        Fused GRU cell: gi and gh already contain projections + biases.
        h_new = (1 - z) * n + z * h
        where r = sigmoid(gi_r + gh_r)
              z = sigmoid(gi_z + gh_z)
              n = tanh(gi_n + r * gh_n)
        """
        pid_b = tl.program_id(0)
        pid_g = tl.program_id(1)

        offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        mask_g = offs_g < G

        # Load h_state
        h_ptrs = h_ptr + pid_b * stride_h_b + offs_g * stride_h_g
        h_state = tl.load(h_ptrs, mask=mask_g, other=0.0).to(tl.float32)

        # Load gi components [0:G], [G:2G], [2G:3G]
        gi_r_ptrs = gi_ptr + pid_b * stride_gi_b + offs_g * stride_gi_g
        gi_z_ptrs = gi_ptr + pid_b * stride_gi_b + (G + offs_g) * stride_gi_g
        gi_n_ptrs = gi_ptr + pid_b * stride_gi_b + (2 * G + offs_g) * stride_gi_g

        gi_r = tl.load(gi_r_ptrs, mask=mask_g, other=0.0).to(tl.float32)
        gi_z = tl.load(gi_z_ptrs, mask=mask_g, other=0.0).to(tl.float32)
        gi_n = tl.load(gi_n_ptrs, mask=mask_g, other=0.0).to(tl.float32)

        # Load gh components
        gh_r_ptrs = gh_ptr + pid_b * stride_gh_b + offs_g * stride_gh_g
        gh_z_ptrs = gh_ptr + pid_b * stride_gh_b + (G + offs_g) * stride_gh_g
        gh_n_ptrs = gh_ptr + pid_b * stride_gh_b + (2 * G + offs_g) * stride_gh_g

        gh_r = tl.load(gh_r_ptrs, mask=mask_g, other=0.0).to(tl.float32)
        gh_z = tl.load(gh_z_ptrs, mask=mask_g, other=0.0).to(tl.float32)
        gh_n = tl.load(gh_n_ptrs, mask=mask_g, other=0.0).to(tl.float32)

        # GRU computations
        r = tl.sigmoid(gi_r + gh_r)
        z = tl.sigmoid(gi_z + gh_z)
        # Triton doesn't have tl.tanh; use tanh(x) = 2*sigmoid(2x) - 1
        n = 2.0 * tl.sigmoid(2.0 * (gi_n + r * gh_n)) - 1.0
        h_new = (1.0 - z) * n + z * h_state

        # Store
        h_out_ptrs = h_out_ptr + pid_b * stride_hout_b + offs_g * stride_hout_g
        tl.store(h_out_ptrs, h_new.to(h_ptr.dtype.element_ty), mask=mask_g)


class DominoGraphRunner:
    """
    CUDA Graph runner for Domino with optional Triton fused per-step projection.

    Uses inline Triton kernels (_v5_fused_gemv_argmax_kernel + _v5_reduce_argmax_kernel)
    to fuse SiLU + fc2 GEMV + base_logits add + argmax into 2 kernel launches per step.

    Also supports handwritten GRUCell + precomputed embedding@W_ih.T for the sequential
    hidden state update, avoiding per-step embed_tokens + nn.GRU(seq_len=1) overhead.

    Assumptions:
        - greedy decoding / argmax
        - fixed batch_size and steps
        - base_logits are precomputed outside
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        batch_size: int,
        steps: int,
        hidden_dim: int,
        gru_hidden_dim: int,
        vocab_size: int,
        prefix_token_count: int,
        device: torch.device,
    ):
        self.steps = steps
        self.draft_model = draft_model
        self.target_model = target_model
        self.prefix_token_count = prefix_token_count
        self.gru_hidden_dim = gru_hidden_dim

        fc1, self.middle, fc2 = unwrap_domino_mlp(draft_model.embed_proj)

        expected_in = hidden_dim + gru_hidden_dim
        if fc1.in_features != expected_in:
            raise ValueError(
                f"v5 fc1 input dim mismatch: got {fc1.in_features}, "
                f"expected {expected_in} = hidden_dim + gru_hidden_dim."
            )

        self.w_z = fc1.weight[:, :hidden_dim].detach().contiguous()
        self.w_s = fc1.weight[:, hidden_dim:].detach().contiguous()
        self.fc1_bias = fc1.bias.detach() if fc1.bias is not None else None

        self.fc2_weight = fc2.weight.detach().contiguous()
        self.fc2_bias = fc2.bias.detach() if fc2.bias is not None else None

        self.use_bias_norm = bool(getattr(draft_model, "use_bias_norm", False))
        self.use_bias_gate = bool(
            getattr(draft_model, "use_bias_gate", False)
            and hasattr(draft_model, "bias_gate")
        )

        # Strict correctness guard: only fuse when middle is exactly SiLU
        self._middle_is_silu = (
            isinstance(self.middle, nn.SiLU)
            or (
                isinstance(self.middle, nn.Sequential)
                and len(list(self.middle.children())) == 1
                and isinstance(list(self.middle.children())[0], nn.SiLU)
            )
        )
        # Enable triton fusion whenever middle is SiLU (bias/gate handled in kernel)
        self._can_triton_fuse = self._middle_is_silu and HAS_TRITON

        dtype = next(draft_model.parameters()).dtype

        # Pre-allocate reduction buffers for triton kernel (used inside graph)
        self._BLOCK_V = 512
        self._num_v_blocks = (vocab_size + self._BLOCK_V - 1) // self._BLOCK_V
        self._triton_out_val = torch.zeros(
            batch_size, self._num_v_blocks,
            dtype=torch.float32, device=device,
        )
        self._triton_out_idx = torch.zeros(
            batch_size, self._num_v_blocks,
            dtype=torch.int32, device=device,
        )
        self._triton_final = torch.zeros(
            batch_size, dtype=torch.int64, device=device,
        )

        # Pre-allocate GRU cell output buffer for fused Triton kernel
        self._gru_h_out = torch.zeros(
            batch_size, gru_hidden_dim,
            dtype=dtype, device=device,
        )

        # ---------- Phase 1: extract GRU weights ----------
        gru = draft_model.prefix_gru
        assert gru.num_layers == 1 and not gru.bidirectional
        self.gru_w_ih = gru.weight_ih_l0.detach().contiguous()
        self.gru_w_hh = gru.weight_hh_l0.detach().contiguous()
        self.gru_b_ih = gru.bias_ih_l0.detach().contiguous() if gru.bias else None
        self.gru_b_hh = gru.bias_hh_l0.detach().contiguous() if gru.bias else None

        # ---------- Phase 2: precompute embedding @ W_ih.T ----------
        # table[cur_token] gives gi directly, skipping embed_tokens + input GEMV
        embed_weight = target_model.model.embed_tokens.weight
        self._gru_input_proj_table = F.linear(
            embed_weight, self.gru_w_ih, self.gru_b_ih
        ).contiguous()

        self.static_prefix_ids = torch.zeros(
            batch_size,
            prefix_token_count,
            dtype=torch.long,
            device=device,
        )

        self.static_parallel_hiddens = torch.zeros(
            batch_size,
            steps,
            hidden_dim,
            dtype=dtype,
            device=device,
        )

        self.static_base_logits = torch.zeros(
            batch_size,
            steps,
            vocab_size,
            dtype=dtype,
            device=device,
        )

        self.static_out = torch.zeros(
            batch_size,
            steps,
            dtype=torch.long,
            device=device,
        )

        # Warmup on side stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.inference_mode(), torch.cuda.stream(s):
            for _ in range(3):
                self._forward_impl()

        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self._forward_impl()

    def _forward_impl(self):
        """
        Graph-captured v5 rollout.
        """

        # Precompute H branch:
        # fc1([z, s]) = z @ W_z.T + s @ W_s.T + bias
        z_part_all = F.linear(
            self.static_parallel_hiddens,
            self.w_z,
            self.fc1_bias,
        )  # [B, S, M]

        # Gate only depends on H, so it can be computed once.
        if self.use_bias_gate:
            gate_all = torch.sigmoid(
                self.draft_model.bias_gate(self.static_parallel_hiddens)
            )  # [B, S, 1]
        else:
            gate_all = None

        prefix_embeds = self.target_model.model.embed_tokens(
            self.static_prefix_ids
        )

        _, gru_hidden = self.draft_model.prefix_gru(prefix_embeds)

        for t in range(self.steps):
            # s_proj directly from gru_hidden [1,B,G] → squeeze → [B,G] → linear → [B,M]
            h_state = gru_hidden.squeeze(0)  # [B, G]

            if self.use_bias_norm:
                h_state = self.draft_model.bias_norm(h_state.unsqueeze(1)).squeeze(1)

            s_proj = F.linear(h_state, self.w_s, None)  # [B, M]

            if self._can_triton_fuse:
                # Fused: SiLU(z[t]+s_proj) @ fc2.T + [bias] + base[t] → [gate *] → argmax
                gate_step = gate_all[:, t, :] if gate_all is not None else None
                cur = self._fused_step_triton(
                    z_part_all[:, t, :],            # [B, M]
                    s_proj,                          # [B, M]
                    self.static_base_logits[:, t, :], # [B, V]
                    gate_val=gate_step,
                ).unsqueeze(-1)  # [B, 1]
            else:
                mid = self.middle(
                    z_part_all[:, t:t + 1, :] + s_proj.unsqueeze(1)
                )  # [B, 1, M]
                bias = F.linear(mid, self.fc2_weight, self.fc2_bias)
                if gate_all is not None:
                    bias = gate_all[:, t:t + 1, :] * bias
                logits = self.static_base_logits[:, t:t + 1, :] + bias
                cur = torch.argmax(logits, dim=-1)  # [B, 1]

            self.static_out[:, t:t + 1] = cur

            if t + 1 < self.steps:
                # Phase 2: gi = embedding @ W_ih.T, gathered by cur token
                gi = self._gru_input_proj_table[cur.squeeze(-1)]  # [B, 3*G]
                # Phase 1: GRUCell hidden update
                gh = F.linear(h_state, self.gru_w_hh, self.gru_b_hh)  # [B, 3*G]
                if self._can_triton_fuse:
                    self._fused_gru_cell(gi, gh, h_state, self._gru_h_out)
                    gru_hidden = self._gru_h_out.unsqueeze(0)  # [1, B, G]
                else:
                    i_r, i_z, i_n = gi.chunk(3, dim=-1)
                    h_r, h_z, h_n = gh.chunk(3, dim=-1)
                    r = torch.sigmoid(i_r + h_r)
                    z = torch.sigmoid(i_z + h_z)
                    n = torch.tanh(i_n + r * h_n)
                    h_new = (1.0 - z) * n + z * h_state
                    gru_hidden = h_new.unsqueeze(0)  # [1, B, G]

    def _fused_step_triton(self, z_proj, s_proj, base_step, gate_val=None):
        """
        Fused: SiLU(z_proj + s_proj) @ fc2_weight.T + [fc2_bias] + base_step → [gate *] → argmax.

        z_proj:   [B, M]
        s_proj:   [B, M]
        base_step: [B, V]
        gate_val:  [B, 1] or None
        returns:  [B]
        """
        B, M = z_proj.shape
        V = base_step.shape[1]

        has_fc2_bias = self.fc2_bias is not None
        has_gate = gate_val is not None

        fc2_bias_ptr = self.fc2_bias if has_fc2_bias else self._triton_out_val
        gate_ptr = gate_val if has_gate else self._triton_out_val

        # Kernel: GEMV + SiLU + add [bias] + base + [gate] + per-block argmax
        grid = (self._num_v_blocks, B)
        _v5_fused_gemv_argmax_kernel[grid](
            z_proj, s_proj,
            self.fc2_weight, base_step, fc2_bias_ptr, gate_ptr,
            self._triton_out_val, self._triton_out_idx,
            M, V,
            z_proj.stride(0), z_proj.stride(1),
            s_proj.stride(0), s_proj.stride(1),
            self.fc2_weight.stride(0), self.fc2_weight.stride(1),
            base_step.stride(0), base_step.stride(1),
            self._triton_out_val.stride(0), self._triton_out_val.stride(1),
            BLOCK_V=self._BLOCK_V,
            BLOCK_M=32,
            HAS_FC2_BIAS=has_fc2_bias,
            HAS_GATE=has_gate,
        )

        # Global argmax reduction
        _v5_reduce_argmax_kernel[(B,)](
            self._triton_out_val, self._triton_out_idx,
            self._triton_final,
            self._num_v_blocks,
            self._triton_out_val.stride(0), self._triton_out_val.stride(1),
            self._triton_out_idx.stride(0), self._triton_out_idx.stride(1),
            self._triton_final.stride(0),
            BLOCK_N=512,
        )
        return self._triton_final

    def _fused_gru_cell(self, gi, gh, h_state, h_out):
        B, G3 = gi.shape
        G = self.gru_hidden_dim

        BLOCK_G = 256
        grid = (B, (G + BLOCK_G - 1) // BLOCK_G)

        _v5_fused_gru_cell_kernel[grid](
            gi, gh, h_state, h_out,
            B, G,
            gi.stride(0), gi.stride(1),
            gh.stride(0), gh.stride(1),
            h_state.stride(0), h_state.stride(1),
            h_out.stride(0), h_out.stride(1),
            BLOCK_G=BLOCK_G,
        )

    @torch.inference_mode()
    def __call__(
        self,
        prefix_ids: torch.Tensor,
        parallel_hiddens: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        self.static_prefix_ids.copy_(prefix_ids)
        self.static_parallel_hiddens.copy_(parallel_hiddens)
        self.static_base_logits.copy_(base_logits)

        self.graph.replay()

        return self.static_out.clone()
