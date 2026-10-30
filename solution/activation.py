"""Собственный Triton-кернел для быстрой активации."""

import torch
import triton
import triton.language as tl


@triton.jit
def custom_gelu_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    # Быстрое приближение GeLU
    cdf = 0.5 * (1.0 + tl.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
    out = x * cdf
    tl.store(out_ptr + idx, out.to(x_ptr.dtype.element_ty), mask=mask)


def custom_activation(x):
    if not x.is_cuda:
        return torch.nn.functional.gelu(x)
    out = torch.empty_like(x)
    n = x.numel()
    block_size = 1024
    grid = (triton.cdiv(n, block_size),)
    custom_gelu_kernel[grid](x, out, n, BLOCK_SIZE=block_size)
    return out