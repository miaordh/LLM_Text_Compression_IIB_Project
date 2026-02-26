import math
import os
import types
import importlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_CPU_THREADS_CONFIGURED = False

try:
    deterministic_ops_cpp = importlib.import_module("deterministic_ops_cpp")
    HAS_DETERMINISTIC_CPP = True
except ImportError:
    deterministic_ops_cpp = None
    HAS_DETERMINISTIC_CPP = False


@dataclass
class DeterministicKernelConfig:
    mode: str = "strict_cpu"
    seed: int = 0
    patch_linear: bool = True
    patch_rmsnorm: bool = True
    patch_attention: bool = True
    force_cpu_kernels: bool = True
    single_thread_cpu: bool = True


class PatchHandle:
    def __init__(self):
        self._restorers = []

    def add_restorer(self, fn):
        self._restorers.append(fn)

    def restore(self):
        while self._restorers:
            fn = self._restorers.pop()
            fn()


def configure_torch_determinism(seed: int = 0, single_thread_cpu: bool = True):
    global _CPU_THREADS_CONFIGURED

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False

    if single_thread_cpu and not _CPU_THREADS_CONFIGURED:
        try:
            torch.set_num_threads(1)
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _CPU_THREADS_CONFIGURED = True


def _cpu_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits_cpu = logits.detach().cpu().to(torch.float64).contiguous()
    max_vals = torch.max(logits_cpu, dim=dim, keepdim=True).values
    exps = torch.exp(logits_cpu - max_vals)
    denom = torch.sum(exps, dim=dim, keepdim=True)
    return exps / denom


def deterministic_softmax(logits: torch.Tensor, dim: int = -1, mode: str = "strict_cpu") -> torch.Tensor:
    if mode == "gpu_best_effort":
        return torch.softmax(logits, dim=dim)

    if dim != -1:
        logits = logits.transpose(dim, -1)
        transposed = True
    else:
        transposed = False

    if HAS_DETERMINISTIC_CPP:
        out_cpu = deterministic_ops_cpp.softmax(logits)
    else:
        out_cpu = _cpu_softmax(logits, dim=-1)

    if transposed:
        out_cpu = out_cpu.transpose(dim, -1)

    return out_cpu.to(device=logits.device, dtype=logits.dtype)


def deterministic_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if HAS_DETERMINISTIC_CPP and a.dim() == 2 and b.dim() == 2:
        out = deterministic_ops_cpp.matmul(a, b)
    else:
        out = torch.matmul(a.detach().cpu().to(torch.float64), b.detach().cpu().to(torch.float64))
    return out.to(device=a.device, dtype=a.dtype)


def deterministic_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if HAS_DETERMINISTIC_CPP:
        out = deterministic_ops_cpp.rmsnorm(x, weight, float(eps))
    else:
        x_cpu = x.detach().cpu().to(torch.float64)
        w_cpu = weight.detach().cpu().to(torch.float64)
        ms = (x_cpu * x_cpu).mean(dim=-1, keepdim=True)
        out = x_cpu * torch.rsqrt(ms + eps) * w_cpu
    return out.to(device=x.device, dtype=x.dtype)


def _patch_linear_modules(model: nn.Module, handle: PatchHandle):
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue

        original_forward = module.forward

        def _det_forward(self, x):
            orig_shape = x.shape
            x2d = x.reshape(-1, self.in_features)
            out2d = deterministic_matmul(x2d, self.weight.t())
            if self.bias is not None:
                out2d = out2d + self.bias
            return out2d.reshape(*orig_shape[:-1], self.out_features)

        module.forward = types.MethodType(_det_forward, module)

        def _restore(m=module, fn=original_forward):
            m.forward = fn

        handle.add_restorer(_restore)


def _patch_rmsnorm_modules(model: nn.Module, handle: PatchHandle):
    for module in model.modules():
        cls_name = module.__class__.__name__.lower()
        if "rmsnorm" not in cls_name:
            continue
        if not hasattr(module, "weight"):
            continue

        eps = getattr(module, "variance_epsilon", None)
        if eps is None:
            eps = getattr(module, "eps", 1e-6)

        original_forward = module.forward

        def _det_forward(self, hidden_states):
            return deterministic_rmsnorm(hidden_states, self.weight, eps)

        module.forward = types.MethodType(_det_forward, module)

        def _restore(m=module, fn=original_forward):
            m.forward = fn

        handle.add_restorer(_restore)


def _deterministic_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    if dropout_p != 0.0:
        raise ValueError("deterministic SDPA requires dropout_p=0.0")

    q = query.detach().cpu().to(torch.float64)
    k = key.detach().cpu().to(torch.float64)
    v = value.detach().cpu().to(torch.float64)

    if enable_gqa and q.size(-3) != k.size(-3):
        ratio = q.size(-3) // k.size(-3)
        k = k.repeat_interleave(ratio, dim=-3)
        v = v.repeat_interleave(ratio, dim=-3)

    d = q.size(-1)
    scale_value = (1.0 / math.sqrt(d)) if scale is None else float(scale)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale_value

    if is_causal:
        q_len = scores.size(-2)
        k_len = scores.size(-1)
        causal = torch.triu(torch.ones((q_len, k_len), dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask.cpu(), float("-inf"))
        else:
            scores = scores + attn_mask.cpu().to(torch.float64)

    probs = deterministic_softmax(scores, dim=-1).cpu().to(torch.float64)
    out = torch.matmul(probs, v)
    return out.to(device=query.device, dtype=query.dtype)


def _patch_attention_function(handle: PatchHandle):
    if not hasattr(F, "scaled_dot_product_attention"):
        return

    original_fn = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _deterministic_sdpa

    def _restore():
        F.scaled_dot_product_attention = original_fn

    handle.add_restorer(_restore)


@contextmanager
def deterministic_kernel_context(model: nn.Module, config: Optional[DeterministicKernelConfig] = None):
    cfg = config or DeterministicKernelConfig()
    configure_torch_determinism(seed=cfg.seed, single_thread_cpu=cfg.single_thread_cpu)

    handle = PatchHandle()

    strict_cpu = (cfg.mode == "strict_cpu") and cfg.force_cpu_kernels

    if cfg.patch_linear and strict_cpu:
        _patch_linear_modules(model, handle)
    if cfg.patch_rmsnorm and strict_cpu:
        _patch_rmsnorm_modules(model, handle)
    if cfg.patch_attention and strict_cpu:
        _patch_attention_function(handle)

    try:
        yield
    finally:
        handle.restore()
