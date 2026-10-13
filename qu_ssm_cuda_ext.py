"""
Loader for the optional QU-SSM fused CUDA forward kernel.

Important: the CUDA kernel in qu_ssm_cuda.cu implements ONLY the forward
recurrence. There is no backward kernel. This module therefore:
  1. Never compiles/loads on import unless CUDA is actually available.
  2. Only ever calls the kernel when gradients are not required (an explicit
     `torch.is_grad_enabled()` / `requires_grad` check happens both here and,
     redundantly, inside the C++ TORCH_CHECK).
  3. Falls back to `None` (caller must fall back to the pure-PyTorch scan)
     on any failure to build/load, so environments without a CUDA toolchain
     are unaffected.

This file has NOT been compiled or run in the environment that produced it
(no GPU / nvcc / torch was available there). Build and test it on your own
CUDA machine before relying on it; the fallback path (pure PyTorch) is the
one that has actually been verified.
"""

import os
import torch

_EXT = None
_LOAD_ATTEMPTED = False
N_HEADS = 8  # must match the constant compiled into qu_ssm_cuda.cu


def _try_load():
    global _EXT, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _EXT
    _LOAD_ATTEMPTED = True

    if not torch.cuda.is_available() or os.environ.get("QU_SSM_NO_CUDA_JIT", "0") == "1":
        return None

    try:
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qu_ssm_cuda.cu")
        if not os.path.exists(src):
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "qu_ssm_cuda.cu")
        _EXT = load(
            name="qu_ssm_cuda_ext",
            sources=[src],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as e:  # noqa: BLE001 - deliberately broad: any build failure -> fallback
        print(f"[qu_ssm_cuda_ext] Could not build/load fused CUDA kernel, "
              f"falling back to pure PyTorch scan. Reason: {e}")
        _EXT = None

    return _EXT


def fused_forward_available(config) -> bool:
    """Whether the fast inference kernel can be used for this config, right now."""
    if getattr(config, "d_state", None) != N_HEADS:
        return False
    if not torch.cuda.is_available():
        return False
    return _try_load() is not None


def fused_forward(u_val, g_val, th_p, theta_base_per_head):
    """
    Inference-only fused forward. Caller (ExactRealQUBlock.forward) is
    responsible for only calling this under torch.no_grad() / eval() and for
    confirming fused_forward_available(config) first.

    Args:
        u_val: [B, L, D] pre-expansion input projection (real part branch)
        g_val: [B, L, D] pre-softplus gate projection (raw, before -softplus)
        th_p:  [B, L, D] pre-offset theta projection (before + theta_base)
        theta_base_per_head: [N_HEADS] float32, one value per state/head

    Returns:
        h_sum: [B, L, D] -- already summed over the state/head axis, matching
        `h_re.sum(dim=-1)` in the pure-PyTorch path.
    """
    ext = _try_load()
    if ext is None:
        raise RuntimeError(
            "fused_forward() called but the CUDA extension is not available. "
            "Call fused_forward_available(config) first."
        )
    if torch.is_grad_enabled() and (u_val.requires_grad or g_val.requires_grad or th_p.requires_grad):
        raise RuntimeError(
            "fused_forward() has no backward pass -- do not call it with "
            "gradients enabled. Wrap the call in torch.no_grad()."
        )
    return ext.fused_fwd(u_val, g_val, th_p, theta_base_per_head.float().contiguous())
