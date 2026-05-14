import importlib
import importlib.util
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    from transformers.cache_utils import DynamicCache

    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False


def normalize_determinism_mode(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    normalized = str(mode).strip().lower()
    if normalized in {"", "none", "off", "false", "0"}:
        return None
    if normalized not in {"batch_invariant_ops", "tbik"}:
        raise ValueError(
            "Unsupported determinism_mode. Expected one of: None, batch_invariant_ops, tbik"
        )
    return normalized


def add_llm_reproducibility_paths() -> bool:
    root = Path(__file__).resolve().parent / "llm_reproducibility"
    src_dir = root / "src"
    added = False
    for candidate in (root, src_dir):
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            added = True
    return added


def import_batch_invariant_backend(prefer_repo_backend: bool = False):
    def _load_from_module(module_name: str):
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            return None, None
        return getattr(mod, "set_batch_invariant_mode", None), getattr(mod, "log_softmax", None)

    if prefer_repo_backend and add_llm_reproducibility_paths():
        set_mode, log_softmax = _load_from_module("bio.batch_invariant_ops")
        if set_mode is not None and log_softmax is not None:
            return set_mode, log_softmax

    set_mode, log_softmax = _load_from_module("batch_invariant_ops")
    if set_mode is not None and log_softmax is not None:
        return set_mode, log_softmax

    set_mode, log_softmax = _load_from_module("batch_invariant_ops.batch_invariant_ops")
    if set_mode is not None and log_softmax is not None:
        return set_mode, log_softmax

    if add_llm_reproducibility_paths():
        set_mode, log_softmax = _load_from_module("bio.batch_invariant_ops")
        if set_mode is not None and log_softmax is not None:
            return set_mode, log_softmax

    return None, None


def apply_tbik_patches():
    if not add_llm_reproducibility_paths():
        raise RuntimeError(
            "determinism_mode='tbik' requires the llm_reproducibility repository at ./llm_reproducibility"
        )

    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["VLLM_TP_INVARIANT"] = "1"

    try:
        patch_module = importlib.import_module("src.patch")
        apply_patches = getattr(patch_module, "apply_patches")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import TBIK patch entrypoint from llm_reproducibility. "
            "Install dependencies for that repo (for example vllm, triton, and optional mini_allreduce)."
        ) from exc

    try:
        apply_patches()
    except Exception as exc:
        raise RuntimeError(
            "Failed to apply TBIK patches. This usually means vLLM/Triton dependencies are missing "
            "or incompatible with the current environment."
        ) from exc


def triton_is_available() -> bool:
    return importlib.util.find_spec("triton") is not None


class VLLMLogitsBackend:
    @staticmethod
    def _is_gpu_utilization_startup_error(exc: BaseException) -> bool:
        text = str(exc).lower()
        return ("free memory on device" in text) and ("gpu memory utilization" in text)

    def __init__(
        self,
        model_id: str,
        tokenizer,
        device: torch.device,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        torch_dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        attention_backend: Optional[str] = None,
        use_v1: Optional[str] = None,
        max_logprobs: Optional[int] = None,
        max_model_len: Optional[int] = None,
    ):
        if attention_backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = str(attention_backend)
        if use_v1 is not None and str(use_v1).strip() != "":
            os.environ["VLLM_USE_V1"] = str(use_v1)

        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:
            raise RuntimeError(
                "vLLM backend requested, but vllm is not importable. Install vllm on CUDA Linux."
            ) from exc

        if device.type != "cuda":
            raise RuntimeError("vLLM backend currently requires CUDA device")

        self._tokenizer = tokenizer
        self._device = device
        self._vocab_size = int(len(tokenizer))
        self._bos_token_id = getattr(tokenizer, "bos_token_id", None)
        self._pad_token_id = getattr(tokenizer, "pad_token_id", None)
        self._max_logprobs = int(max_logprobs) if max_logprobs is not None else int(self._vocab_size)
        if self._max_logprobs <= 0:
            self._max_logprobs = int(self._vocab_size)

        requested_tp_size = max(1, int(tensor_parallel_size))
        visible_gpu_count = int(torch.cuda.device_count())
        if requested_tp_size > visible_gpu_count:
            raise RuntimeError(
                f"vLLM tensor_parallel_size={requested_tp_size} requires at least "
                f"{requested_tp_size} visible CUDA devices, but torch sees {visible_gpu_count}. "
                "Set CUDA_VISIBLE_DEVICES to include enough GPUs, for example CUDA_VISIBLE_DEVICES=0,1."
            )

        engine_kwargs: Dict[str, Any] = {
            "model": model_id,
            "trust_remote_code": bool(trust_remote_code),
            "tensor_parallel_size": requested_tp_size,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "max_logprobs": self._max_logprobs,
            "logprobs_mode": "raw_logprobs",
        }
        if revision:
            engine_kwargs["revision"] = revision

        if torch_dtype in {"float16", "bfloat16", "float32"}:
            engine_kwargs["dtype"] = torch_dtype
        if max_model_len is not None and int(max_model_len) > 0:
            engine_kwargs["max_model_len"] = int(max_model_len)

        requested_gpu_util = float(gpu_memory_utilization)
        util_candidates: List[float] = []
        for candidate in (requested_gpu_util, min(requested_gpu_util, 0.90), 0.85, 0.80):
            if 0.50 <= float(candidate) <= 0.99 and all(abs(candidate - u) > 1e-9 for u in util_candidates):
                util_candidates.append(float(candidate))

        self._llm = None
        last_exc = None
        for util in util_candidates:
            try:
                engine_kwargs["gpu_memory_utilization"] = float(util)
                self._llm = LLM(**engine_kwargs)
                break
            except Exception as exc:
                last_exc = exc
                if not self._is_gpu_utilization_startup_error(exc):
                    raise
                continue

        if self._llm is None:
            tried = ", ".join(f"{u:.2f}" for u in util_candidates)
            if last_exc is not None:
                raise RuntimeError(
                    f"vLLM engine initialization failed after retrying gpu_memory_utilization values: {tried}"
                ) from last_exc
            raise RuntimeError("vLLM engine initialization failed with unknown error")

        self._sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            seed=1,
            logprobs=self._max_logprobs,
            detokenize=False,
            ignore_eos=True,
        )

    def _prompt_ids(self, prefix_ids: Sequence[int]) -> List[int]:
        if prefix_ids:
            return [int(t) for t in prefix_ids]
        bos = self._bos_token_id
        if bos is None:
            bos = self._pad_token_id
        if bos is None:
            bos = 0
        return [int(bos)]

    def _logprob_value(self, value: Any) -> float:
        if hasattr(value, "logprob"):
            return float(value.logprob)
        return float(value)

    def _logits_from_step_dict(self, step_dict: Dict[Any, Any]) -> torch.Tensor:
        probs = np.zeros((self._vocab_size,), dtype=np.float64)
        assigned = np.zeros((self._vocab_size,), dtype=np.bool_)

        for token_id, payload in step_dict.items():
            idx = int(token_id)
            if 0 <= idx < self._vocab_size:
                lp = self._logprob_value(payload)
                p = float(np.exp(lp))
                if p > 0.0:
                    probs[idx] = p
                    assigned[idx] = True

        known_mass = float(probs.sum())
        unknown = int((~assigned).sum())
        residual = max(0.0, 1.0 - known_mass)
        if unknown > 0:
            fill = residual / float(unknown)
            if fill <= 0.0:
                fill = 1e-12
            probs[~assigned] = fill

        total = float(probs.sum())
        if not np.isfinite(total) or total <= 0.0:
            probs[:] = 1.0 / float(self._vocab_size)
        else:
            probs /= total

        logits = np.log(np.clip(probs, 1e-45, 1.0)).astype(np.float32, copy=False)
        return torch.from_numpy(logits).to(device=self._device)

    def _generate_for_prompt_ids(self, prompt_ids_list: Sequence[Sequence[int]]):
        prompt_ids_list = [[int(token_id) for token_id in prompt_ids] for prompt_ids in prompt_ids_list]
        outputs = None
        last_type_error = None

        generate_attempts = [
            lambda: self._llm.generate(
                prompt_token_ids=prompt_ids_list,
                sampling_params=self._sampling_params,
                use_tqdm=False,
            ),
            lambda: self._llm.generate(
                prompt=prompt_ids_list,
                sampling_params=self._sampling_params,
                use_tqdm=False,
            ),
            lambda: self._llm.generate(
                prompts=prompt_ids_list,
                sampling_params=self._sampling_params,
                use_tqdm=False,
            ),
            lambda: self._llm.generate(
                [{"prompt_token_ids": prompt_ids} for prompt_ids in prompt_ids_list],
                self._sampling_params,
                use_tqdm=False,
            ),
            lambda: self._llm.generate(
                prompt_ids_list,
                self._sampling_params,
                use_tqdm=False,
            ),
        ]

        for attempt in generate_attempts:
            try:
                outputs = attempt()
                break
            except TypeError as exc:
                last_type_error = exc
                continue

        if outputs is None:
            if last_type_error is not None:
                raise last_type_error
            raise RuntimeError("vLLM generate returned no result and no TypeError was captured")
        return outputs

    def next_logits(self, prefix_ids: Sequence[int]) -> torch.Tensor:
        prompt_ids = self._prompt_ids(prefix_ids)
        return self.next_logits_batch([prompt_ids], already_prompt_ids=True)[0]

    def next_logits_batch(
        self,
        prefix_ids_list: Sequence[Sequence[int]],
        already_prompt_ids: bool = False,
    ) -> List[torch.Tensor]:
        if already_prompt_ids:
            prompt_ids_list = [[int(token_id) for token_id in prefix_ids] for prefix_ids in prefix_ids_list]
        else:
            prompt_ids_list = [self._prompt_ids(prefix_ids) for prefix_ids in prefix_ids_list]

        outputs = self._generate_for_prompt_ids(prompt_ids_list)
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned empty outputs while requesting next-token logits")
        if len(outputs) != len(prompt_ids_list):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(prompt_ids_list)} prompts"
            )

        logits_list: List[torch.Tensor] = []
        for output in outputs:
            if not output.outputs:
                raise RuntimeError("vLLM returned an empty per-prompt output")
            token_logprobs = output.outputs[0].logprobs
            if not token_logprobs:
                raise RuntimeError("vLLM did not return token logprobs for the next token")
            logits_list.append(self._logits_from_step_dict(token_logprobs[0]))
        return logits_list


def cpu_batch_invariant_log_softmax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if dim < 0:
        dim += input_tensor.ndim
    if dim < 0 or dim >= input_tensor.ndim:
        raise ValueError("Invalid dim for log_softmax")

    x = input_tensor
    moved = False
    if dim != x.ndim - 1:
        x = x.movedim(dim, -1)
        moved = True

    original_shape = x.shape
    n_cols = int(original_shape[-1])
    x2 = x.reshape(-1, n_cols)
    x32 = x2.to(torch.float32)

    block_size = 1024
    max_vals = torch.full((x32.shape[0],), -float("inf"), dtype=torch.float32, device=x32.device)
    for start in range(0, n_cols, block_size):
        block = x32[:, start : start + block_size]
        max_vals = torch.maximum(max_vals, block.max(dim=1).values)

    sum_exp = torch.zeros((x32.shape[0],), dtype=torch.float32, device=x32.device)
    for start in range(0, n_cols, block_size):
        block = x32[:, start : start + block_size]
        sum_exp = sum_exp + torch.exp(block - max_vals[:, None]).sum(dim=1)

    out32 = x32 - max_vals[:, None] - torch.log(sum_exp)[:, None]
    out = out32.to(x2.dtype).reshape(original_shape)

    if moved:
        out = out.movedim(-1, dim)
    return out


def configure_determinism_runtime(device: torch.device, determinism_mode: Optional[str]):
    """Return (enabled, context_manager_factory, log_softmax_fn) for runtime determinism."""
    batch_invariant_enabled = False
    batch_invariant_ctx = lambda _enabled=True: nullcontext()
    log_softmax_fn = torch.log_softmax

    if determinism_mode is None:
        return batch_invariant_enabled, batch_invariant_ctx, log_softmax_fn

    if device.type == "cpu":
        batch_invariant_enabled = True
        batch_invariant_ctx = lambda _enabled=True: nullcontext()
        log_softmax_fn = cpu_batch_invariant_log_softmax

        try:
            cpu_ops_path = str(Path(__file__).resolve().parent / "cpu_batch_invariant_ops")
            if cpu_ops_path not in sys.path:
                sys.path.append(cpu_ops_path)
            from patch_cpu_determinism import patch_llama_for_cpu_determinism

            patch_llama_for_cpu_determinism()
        except Exception as e:
            print(f"Warning: Failed to initialize CPU deep deterministic patch: {e}")

        return batch_invariant_enabled, batch_invariant_ctx, log_softmax_fn

    if determinism_mode == "tbik":
        if device.type != "cuda":
            batch_invariant_enabled = True
            batch_invariant_ctx = lambda _enabled=True: nullcontext()
            log_softmax_fn = cpu_batch_invariant_log_softmax

            try:
                cpu_ops_path = str(Path(__file__).resolve().parent / "cpu_batch_invariant_ops")
                if cpu_ops_path not in sys.path:
                    sys.path.append(cpu_ops_path)
                from patch_cpu_determinism import patch_llama_for_cpu_determinism

                patch_llama_for_cpu_determinism()
            except Exception as e:
                print(f"Warning: Failed to initialize CPU deep deterministic patch: {e}")

            return batch_invariant_enabled, batch_invariant_ctx, log_softmax_fn

        apply_tbik_patches()

    prefer_repo_backend = determinism_mode == "tbik"
    set_batch_invariant_mode, log_softmax = import_batch_invariant_backend(
        prefer_repo_backend=prefer_repo_backend
    )
    if set_batch_invariant_mode is None or log_softmax is None:
        raise RuntimeError(
            f"determinism_mode='{determinism_mode}' requires batch invariant ops, but no backend could be imported"
        )

    sample_device = device if device.type in {"cuda", "cpu", "mps"} else torch.device("cpu")
    try:
        sample = torch.zeros((1, 4), dtype=torch.float32, device=sample_device)
        with set_batch_invariant_mode(True):
            _ = log_softmax(sample, dim=-1)
        batch_invariant_enabled = True
        batch_invariant_ctx = set_batch_invariant_mode
        log_softmax_fn = log_softmax
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize determinism_mode='{determinism_mode}' backend: {exc}"
        ) from exc

    return batch_invariant_enabled, batch_invariant_ctx, log_softmax_fn
