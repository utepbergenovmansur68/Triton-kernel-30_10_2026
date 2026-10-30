"""Custom Triton kernel for 4-bit NF4 weight unpacking (dequantization)."""

import torch
import triton
import triton.language as tl


@triton.jit
def nf4_dequant_kernel(
    packed_ptr, absmax_ptr, code_ptr, out_ptr,
    n_bytes,
    HALF_BLOCKSIZE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    pid = tl.program_id(0)
    byte_offs = pid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    mask = byte_offs < n_bytes

    packed_byte = tl.load(packed_ptr + byte_offs, mask=mask, other=0).to(tl.int32)
    hi_idx = (packed_byte >> 4) & 0x0F
    lo_idx = packed_byte & 0x0F

    val_hi = tl.load(code_ptr + hi_idx).to(tl.float32)
    val_lo = tl.load(code_ptr + lo_idx).to(tl.float32)

    scale_idx = byte_offs // HALF_BLOCKSIZE
    scale = tl.load(absmax_ptr + scale_idx, mask=mask, other=0.0).to(tl.float32)

    out_hi = (val_hi * scale).to(out_ptr.dtype.element_ty)
    out_lo = (val_lo * scale).to(out_ptr.dtype.element_ty)

    paired = tl.join(out_hi, out_lo)
    out_offs = (pid * BLOCK_BYTES * 2) + tl.arange(0, BLOCK_BYTES * 2)
    out_mask = out_offs < (n_bytes * 2)
    tl.store(out_ptr + tl.reshape(out_offs, (BLOCK_BYTES, 2)), paired, mask=tl.reshape(out_mask, (BLOCK_BYTES, 2)))


def triton_dequantize_nf4(A, quant_state, out=None):
    import bitsandbytes.functional as bnb_F

    if quant_state.nested:
        absmax = bnb_F.dequantize_blockwise(quant_state.absmax, quant_state.state2)
        absmax += quant_state.offset
        if absmax.dtype != torch.float32:
            absmax = absmax.float()
    else:
        absmax = quant_state.absmax
        if absmax.dtype != torch.float32:
            absmax = absmax.float()

    if out is None:
        out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=A.device)

    n_bytes = A.numel()
    half_blocksize = quant_state.blocksize // 2
    block_bytes = 512
    grid = (triton.cdiv(n_bytes, block_bytes),)

    nf4_dequant_kernel[grid](
        A, absmax, quant_state.code, out,
        n_bytes,
        HALF_BLOCKSIZE=half_blocksize,
        BLOCK_BYTES=block_bytes,
        num_warps=4,
    )
    return out


def patch_bnb_nf4():
    import bitsandbytes.functional as bnb_F

    orig_dequantize_4bit = bnb_F.dequantize_4bit

    def custom_dequantize_4bit(
        A, quant_state=None, absmax=None, out=None, blocksize=64, quant_type="fp4"
    ):
        if (
            quant_state is not None
            and A.is_cuda
            and quant_state.blocksize in (64, 128, 256)
            and A.numel() * 2 == quant_state.shape.numel()
        ):
            return triton_dequantize_nf4(A, quant_state, out=out)
        return orig_dequantize_4bit(
            A, quant_state=quant_state, absmax=absmax, out=out,
            blocksize=blocksize, quant_type=quant_type,
        )

    bnb_F.dequantize_4bit = custom_dequantize_4bit