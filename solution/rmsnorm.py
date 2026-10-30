"""Собственный слитый Triton-кернел для Qwen2RMSNorm (forward + backward)."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional dependency for CUDA kernels
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def rmsnorm_fwd_kernel(
        x_ptr, w_ptr, out_ptr, rstd_ptr,
        stride_x_row, stride_out_row,
        n_cols, eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        x_row_ptr = x_ptr + row_idx * stride_x_row
        out_row_ptr = out_ptr + row_idx * stride_out_row

        # Загружаем строку и считаем норму в fp32 прямо в быстрой памяти ядра
        x = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / n_cols
        rstd = tl.rsqrt(var + eps)

        # Сохраняем обратное стандартное отклонение для быстрого backward-прохода
        tl.store(rstd_ptr + row_idx, rstd)

        out = (x * rstd).to(out_ptr.dtype.element_ty).to(tl.float32) * w
        tl.store(out_row_ptr + col_offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


    @triton.jit
    def rmsnorm_bwd_kernel(
        grad_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr,
        stride_row, n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        offset = row_idx * stride_row + col_offsets
        grad = tl.load(grad_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)

        x_hat = x * rstd
        m = grad * w
        c = tl.sum(m * x_hat, axis=0) / n_cols
        dx = rstd * (m - x_hat * c)

        tl.store(dx_ptr + offset, dx.to(dx_ptr.dtype.element_ty), mask=mask)
else:
    def rmsnorm_fwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to run fused RMSNorm kernels.")

    def rmsnorm_bwd_kernel(*args, **kwargs):
        raise ImportError("Triton is required to run fused RMSNorm kernels.")


class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x_2d = x.contiguous().view(-1, n_cols)
        weight = weight.contiguous()
        n_rows = x_2d.shape[0]

        out = torch.empty_like(x_2d)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)

        block_size = triton.next_power_of_2(n_cols)
        rmsnorm_fwd_kernel[(n_rows,)](
            x_2d, weight, out, rstd,
            x_2d.stride(0), out.stride(0),
            n_cols, eps,
            BLOCK_SIZE=block_size,
            num_warps=4 if block_size <= 2048 else 8,
        )
        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.block_size = block_size
        ctx.orig_shape = orig_shape
        return out.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_out):
        x_2d, weight, rstd = ctx.saved_tensors
        n_rows, n_cols = x_2d.shape
        grad_2d = grad_out.contiguous().view(n_rows, n_cols)
        dx = torch.empty_like(x_2d)

        rmsnorm_bwd_kernel[(n_rows,)](
            grad_2d, x_2d, weight, rstd, dx,
            x_2d.stride(0), n_cols,
            BLOCK_SIZE=ctx.block_size,
            num_warps=4 if ctx.block_size <= 2048 else 8,
        )

        dw = None
        if ctx.needs_input_grad[1]:
            x_hat = x_2d.float() * rstd.unsqueeze(1)
            dw = (grad_2d.float() * x_hat).sum(dim=0).to(weight.dtype)

        return dx.view(ctx.orig_shape), dw, None


def rmsnorm(x, weight, eps=1e-6):
    return FusedRMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        if not hidden_states.is_cuda:
            return original(self, hidden_states)
        return rmsnorm(hidden_states, self.weight, self.variance_epsilon)

    modeling_qwen2.Qwen2RMSNorm.forward = forward