"""Второй собственный Triton-кернел для поэлементного масштабирования."""

import torch
import triton
import triton.language as tl


@triton.jit
def scale_kernel(
    x_ptr, out_ptr, scale_factor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    out = x * scale_factor
    tl.store(out_ptr + idx, out.to(x_ptr.dtype.element_ty), mask=mask)


def custom_scale(x, scale_factor=1.0):
    if not x.is_cuda:
        return x * scale_factor
    out = torch.empty_like(x)
    n = x.numel()
    block_size = 1024
    grid = (triton.cdiv(n, block_size),)
    scale_kernel[grid](
        x, out, scale_factor,
        n,
        BLOCK_SIZE=block_size,
    )
    return out