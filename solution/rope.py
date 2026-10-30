"""Fused Rotary Position Embedding (RoPE) forward and backward in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit
def rope_fwd_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    seq_len, n_heads, head_dim, half_dim, cos_bs,
    BLOCK_HALF: tl.constexpr,
):
    row_idx = tl.program_id(0)
    s_idx = row_idx % seq_len
    b_idx = row_idx // (n_heads * seq_len)
    b_cos = 0 if cos_bs == 1 else b_idx

    x_offset = row_idx * head_dim
    cos_offset = (b_cos * seq_len + s_idx) * head_dim

    cols1 = tl.arange(0, BLOCK_HALF)
    mask = cols1 < half_dim
    cols2 = cols1 + half_dim

    x1 = tl.load(x_ptr + x_offset + cols1, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + x_offset + cols2, mask=mask, other=0.0).to(tl.float32)

    c1 = tl.load(cos_ptr + cos_offset + cols1, mask=mask, other=0.0).to(tl.float32)
    c2 = tl.load(cos_ptr + cos_offset + cols2, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(sin_ptr + cos_offset + cols1, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(sin_ptr + cos_offset + cols2, mask=mask, other=0.0).to(tl.float32)

    out1 = x1 * c1 - x2 * s1
    out2 = x2 * c2 + x1 * s2

    tl.store(out_ptr + x_offset + cols1, out1.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + x_offset + cols2, out2.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def rope_bwd_kernel(
    dy_ptr, cos_ptr, sin_ptr, dx_ptr,
    seq_len, n_heads, head_dim, half_dim, cos_bs,
    BLOCK_HALF: tl.constexpr,
):
    row_idx = tl.program_id(0)
    s_idx = row_idx % seq_len
    b_idx = row_idx // (n_heads * seq_len)
    b_cos = 0 if cos_bs == 1 else b_idx

    x_offset = row_idx * head_dim
    cos_offset = (b_cos * seq_len + s_idx) * head_dim

    cols1 = tl.arange(0, BLOCK_HALF)
    mask = cols1 < half_dim
    cols2 = cols1 + half_dim

    dy1 = tl.load(dy_ptr + x_offset + cols1, mask=mask, other=0.0).to(tl.float32)
    dy2 = tl.load(dy_ptr + x_offset + cols2, mask=mask, other=0.0).to(tl.float32)

    c1 = tl.load(cos_ptr + cos_offset + cols1, mask=mask, other=0.0).to(tl.float32)
    c2 = tl.load(cos_ptr + cos_offset + cols2, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(sin_ptr + cos_offset + cols1, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(sin_ptr + cos_offset + cols2, mask=mask, other=0.0).to(tl.float32)

    dx1 = dy1 * c1 + dy2 * s2
    dx2 = dy2 * c2 - dy1 * s1

    tl.store(dx_ptr + x_offset + cols1, dx1.to(dx_ptr.dtype.element_ty), mask=mask)
    tl.store(dx_ptr + x_offset + cols2, dx2.to(dx_ptr.dtype.element_ty), mask=mask)


class FusedRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        q_c = q.contiguous()
        k_c = k.contiguous()
        cos_3d = (cos if cos.ndim == 3 else cos.unsqueeze(0)).contiguous()
        sin_3d = (sin if sin.ndim == 3 else sin.unsqueeze(0)).contiguous()

        bs, n_heads_q, seq_len, head_dim = q_c.shape
        n_heads_k = k_c.shape[1]
        half_dim = head_dim // 2
        block_half = triton.next_power_of_2(half_dim)
        cos_bs = cos_3d.shape[0]

        q_out = torch.empty_like(q_c)
        k_out = torch.empty_like(k_c)

        rope_fwd_kernel[(bs * n_heads_q * seq_len,)](
            q_c, cos_3d, sin_3d, q_out,
            seq_len, n_heads_q, head_dim, half_dim, cos_bs,
            BLOCK_HALF=block_half, num_warps=2,
        )
        rope_fwd_kernel[(bs * n_heads_k * seq_len,)](
            k_c, cos_3d, sin_3d, k_out,
            seq_len, n_heads_k, head_dim, half_dim, cos_bs,
            BLOCK_HALF=block_half, num_warps=2,
        )

        ctx.save_for_backward(cos_3d, sin_3d)
        ctx.shapes = (bs, n_heads_q, n_heads_k, seq_len, head_dim, half_dim, block_half, cos_bs)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq_out, dk_out):
        cos_3d, sin_3d = ctx.saved_tensors
        bs, n_heads_q, n_heads_k, seq_len, head_dim, half_dim, block_half, cos_bs = ctx.shapes

        dq_c = dq_out.contiguous()
        dk_c = dk_out.contiguous()
        dq = torch.empty_like(dq_c)
        dk = torch.empty_like(dk_c)

        rope_bwd_kernel[(bs * n_heads_q * seq_len,)](
            dq_c, cos_3d, sin_3d, dq,
            seq_len, n_heads_q, head_dim, half_dim, cos_bs,
            BLOCK_HALF=block_half, num_warps=2,
        )
        rope_bwd_kernel[(bs * n_heads_k * seq_len,)](
            dk_c, cos_3d, sin_3d, dk,
            seq_len, n_heads_k, head_dim, half_dim, cos_bs,
            BLOCK_HALF=block_half, num_warps=2,
        )
        return dq, dk, None, None


def fused_rope(q, k, cos, sin):
    return FusedRoPE.apply(q, k, cos, sin)


def patch_qwen2_rope():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if not q.is_cuda or unsqueeze_dim != 1 or q.ndim != 4:
            return original(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)
        return fused_rope(q, k, cos, sin)

    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb