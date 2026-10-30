"""Fused Online Softmax + Cross-Entropy loss in Triton for large vocabularies."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, target_ptr, loss_ptr, lse_ptr,
    stride_logits_row, n_cols, ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    target = tl.load(target_ptr + row_idx)
    if target == ignore_index:
        tl.store(loss_ptr + row_idx, 0.0)
        tl.store(lse_ptr + row_idx, 0.0)
        return

    row_start_ptr = logits_ptr + row_idx * stride_logits_row

    max_val = -float("inf")
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(row_start_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_val, block_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
        max_val = new_max

    lse = max_val + tl.log(sum_exp)
    target_logit = tl.load(row_start_ptr + target).to(tl.float32)
    loss = lse - target_logit

    tl.store(loss_ptr + row_idx, loss)
    tl.store(lse_ptr + row_idx, lse)


@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, target_ptr, lse_ptr, grad_row_ptr, dlogits_ptr,
    stride_logits_row, stride_dlogits_row,
    n_cols, ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    cols = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    target = tl.load(target_ptr + row_idx)
    dlogits_row_ptr = dlogits_ptr + row_idx * stride_dlogits_row

    if target == ignore_index:
        tl.store(dlogits_row_ptr + cols, 0.0, mask=mask)
        return

    logits_row_ptr = logits_ptr + row_idx * stride_logits_row
    lse = tl.load(lse_ptr + row_idx).to(tl.float32)
    grad = tl.load(grad_row_ptr + row_idx).to(tl.float32)

    x = tl.load(logits_row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
    probs = tl.exp(x - lse)
    probs = tl.where(cols == target, probs - 1.0, probs)
    dx = probs * grad

    tl.store(dlogits_row_ptr + cols, dx.to(dlogits_ptr.dtype.element_ty), mask=mask)


class FusedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, ignore_index=-100, reduction="mean", return_fp32=False):
        if logits.stride(-1) != 1:
            logits = logits.contiguous()
        target = target.contiguous()
        n_rows, n_cols = logits.shape

        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        block_size = 2048
        cross_entropy_fwd_kernel[(n_rows,)](
            logits, target, losses, lse,
            logits.stride(0), n_cols, ignore_index,
            BLOCK_SIZE=block_size, num_warps=4,
        )

        ctx.save_for_backward(logits, target, lse)
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.block_size = block_size

        out_dtype = torch.float32 if return_fp32 else logits.dtype
        if reduction == "none":
            return losses.to(out_dtype)
        elif reduction == "sum":
            return losses.sum().to(out_dtype)
        else:
            valid = (target != ignore_index).sum()
            if valid == 0:
                return (losses.sum() * 0.0).to(out_dtype)
            return (losses.sum() / valid).to(out_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        logits, target, lse = ctx.saved_tensors
        n_rows, n_cols = logits.shape
        ignore_index = ctx.ignore_index
        reduction = ctx.reduction
        block_size = ctx.block_size

        if reduction == "none":
            grad_row = grad_output.contiguous().view(n_rows).float()
        elif reduction == "sum":
            grad_row = grad_output.float().expand(n_rows).contiguous()
        else:
            valid = (target != ignore_index).sum().clamp(min=1).float()
            grad_row = (grad_output.float() / valid).expand(n_rows).contiguous()

        dlogits = torch.empty_like(logits)
        grid = (n_rows, triton.cdiv(n_cols, block_size))
        cross_entropy_bwd_kernel[grid](
            logits, target, lse, grad_row, dlogits,
            logits.stride(0), dlogits.stride(0),
            n_cols, ignore_index,
            BLOCK_SIZE=block_size, num_warps=4,
        )
        return dlogits, None, None, None, None


def fused_cross_entropy(logits, target, ignore_index=-100, reduction="mean", return_fp32=False):
    return FusedCrossEntropy.apply(logits, target, ignore_index, reduction, return_fp32)


def patch_cross_entropy():
    original_ce = F.cross_entropy

    def custom_cross_entropy(
        input, target, weight=None, size_average=None,
        ignore_index=-100, reduce=None, reduction="mean", label_smoothing=0.0
    ):
        if (
            input.is_cuda
            and target.is_cuda
            and input.ndim == 2
            and target.ndim == 1
            and weight is None
            and label_smoothing == 0.0
            and size_average is None
            and reduce is None
            and reduction in ("mean", "sum", "none")
            and target.dtype == torch.long
        ):
            return fused_cross_entropy(input, target, ignore_index=ignore_index, reduction=reduction)
        return original_ce(
            input, target, weight=weight, size_average=size_average,
            ignore_index=ignore_index, reduce=reduce,
            reduction=reduction, label_smoothing=label_smoothing,
        )

    F.cross_entropy = custom_cross_entropy
    torch.nn.functional.cross_entropy = custom_cross_entropy

    try:
        from transformers.loss import loss_utils
        from transformers.models.qwen2 import modeling_qwen2

        orig_causal_loss = loss_utils.ForCausalLMLoss

        def fast_causal_lm_loss(
            logits, labels, vocab_size: int,
            num_items_in_batch=None, ignore_index: int = -100,
            shift_labels=None, **kwargs
        ):
            if not logits.is_cuda:
                return orig_causal_loss(
                    logits, labels, vocab_size=vocab_size,
                    num_items_in_batch=num_items_in_batch,
                    ignore_index=ignore_index, shift_labels=shift_labels, **kwargs
                )
            if shift_labels is None:
                shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
                logits_2d = logits.contiguous().view(-1, vocab_size)
                labels_1d = shift_labels.view(-1)
            else:
                logits_2d = logits[..., :-1, :].contiguous().view(-1, vocab_size)
                labels_1d = shift_labels.contiguous().view(-1)

            if num_items_in_batch is not None:
                loss = fused_cross_entropy(
                    logits_2d, labels_1d, ignore_index=ignore_index,
                    reduction="sum", return_fp32=True
                )
                return loss / num_items_in_batch
            return fused_cross_entropy(
                logits_2d, labels_1d, ignore_index=ignore_index,
                reduction="mean", return_fp32=True
            )

        loss_utils.ForCausalLMLoss = fast_causal_lm_loss
        if hasattr(loss_utils, "LOSS_MAPPING") and "ForCausalLM" in loss_utils.LOSS_MAPPING:
            loss_utils.LOSS_MAPPING["ForCausalLM"] = fast_causal_lm_loss
        if hasattr(modeling_qwen2.Qwen2PreTrainedModel, "loss_function"):
            modeling_qwen2.Qwen2ForCausalLM.loss_function = staticmethod(fast_causal_lm_loss)
    except Exception:
        pass