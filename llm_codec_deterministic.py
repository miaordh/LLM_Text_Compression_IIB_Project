import gc
import csv
import importlib
import importlib.util
import math
import os
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from arithmetic_coding import Coder
from bitReadWrite import BitReader, BitWriter
from decoder import Decoder
from encoder import Encoder


def _load_project_utils_module():
    module_name = "_project_utils_local"
    if module_name in sys.modules:
        return sys.modules[module_name]

    utils_path = Path(__file__).resolve().with_name("utils.py")
    spec = importlib.util.spec_from_file_location(module_name, str(utils_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load project utils module at {utils_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


_project_utils = _load_project_utils_module()
counts_to_cum_desc = _project_utils.counts_to_cum_desc
probs_to_counts = _project_utils.probs_to_counts
probs_to_counts_legacy = _project_utils.probs_to_counts_legacy

try:
    from transformers.cache_utils import DynamicCache

    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False


def _normalize_determinism_mode(mode: Optional[str]) -> Optional[str]:
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


def _add_llm_reproducibility_paths() -> bool:
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


def _import_batch_invariant_backend(prefer_repo_backend: bool = False):
    def _load_from_module(module_name: str):
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            return None, None
        return getattr(mod, "set_batch_invariant_mode", None), getattr(mod, "log_softmax", None)

    if prefer_repo_backend and _add_llm_reproducibility_paths():
        set_mode, log_softmax = _load_from_module("bio.batch_invariant_ops")
        if set_mode is not None and log_softmax is not None:
            return set_mode, log_softmax

    set_mode, log_softmax = _load_from_module("batch_invariant_ops")
    if set_mode is not None and log_softmax is not None:
        return set_mode, log_softmax

    set_mode, log_softmax = _load_from_module("batch_invariant_ops.batch_invariant_ops")
    if set_mode is not None and log_softmax is not None:
        return set_mode, log_softmax

    if _add_llm_reproducibility_paths():
        set_mode, log_softmax = _load_from_module("bio.batch_invariant_ops")
        if set_mode is not None and log_softmax is not None:
            return set_mode, log_softmax

    return None, None


def _apply_tbik_patches():
    if not _add_llm_reproducibility_paths():
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


def _triton_is_available() -> bool:
    return importlib.util.find_spec("triton") is not None


class _VLLMLogitsBackend:
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
        max_logprobs: Optional[int] = None,
    ):
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

        engine_kwargs: Dict[str, Any] = {
            "model": model_id,
            "trust_remote_code": bool(trust_remote_code),
            "tensor_parallel_size": max(1, int(tensor_parallel_size)),
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

        self._llm = LLM(**engine_kwargs)
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

    def next_logits(self, prefix_ids: Sequence[int]) -> torch.Tensor:
        prompt_ids = self._prompt_ids(prefix_ids)

        outputs = None
        last_type_error = None

        # vLLM API has changed across releases; try common signatures in order.
        generate_attempts = [
            lambda: self._llm.generate(
                prompt_token_ids=[prompt_ids],
                sampling_params=self._sampling_params,
                use_tqdm=False,
            ),
            lambda: self._llm.generate(
                prompt=[prompt_ids],
                sampling_params=self._sampling_params,
                use_tqdm=False,
            ),
            lambda: self._llm.generate(
                prompts=[prompt_ids],
                sampling_params=self._sampling_params,
                use_tqdm=False,
            ),
            lambda: self._llm.generate(
                [{"prompt_token_ids": prompt_ids}],
                self._sampling_params,
                use_tqdm=False,
            ),
            lambda: self._llm.generate(
                [prompt_ids],
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

        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned empty outputs while requesting next-token logits")

        token_logprobs = outputs[0].outputs[0].logprobs
        if not token_logprobs:
            raise RuntimeError("vLLM did not return token logprobs for the next token")

        step_dict = token_logprobs[0]
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


def _cpu_batch_invariant_log_softmax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Fixed-order two-pass reduction on CPU to reduce sensitivity to reduction-order changes.
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


@dataclass
class DeterministicCodecConfig:
    precision: int = 32
    slots: int = (1 << 24)

    context_window: int = 2048
    margin: int = 128
    strategy: str = "rolling"  # rolling | block | no_kv_cache
    use_legacy_counts: bool = False

    quant: bool = False
    logit_round_decimals: int = 2
    prob_round_decimals: int = 5

    # Optional diagnostics CSV prefix/path. When set, codec writes
    # per-token diagnostics to <prefix>_encode.csv and <prefix>_decode.csv.
    diagnostics_csv_prefix: Optional[str] = None

    determinism_mode: Optional[str] = None
    inference_backend: str = "auto"  # auto | huggingface | vllm
    model_id: Optional[str] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None
    torch_dtype: str = "auto"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_logprobs: Optional[int] = None


class DeterministicLLMCodec:
    def __init__(
        self,
        tokenizer,
        model=None,
        device: str = "auto",
        config: Optional[DeterministicCodecConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.config = config or DeterministicCodecConfig()

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self._determinism_mode = _normalize_determinism_mode(self.config.determinism_mode)
        self._inference_backend = self._resolve_inference_backend()

        self.model = None
        self._vllm_backend = None
        if self._inference_backend == "huggingface":
            if model is None:
                raise ValueError("Hugging Face backend requires a loaded model instance")
            self.model = model.to(self.device)
            self.model.eval()

        if self.config.strategy not in {"rolling", "block", "no_kv_cache"}:
            raise ValueError(
                "Unsupported strategy. Expected one of: rolling, block, no_kv_cache"
            )
        if self.config.context_window <= 0:
            raise ValueError("context_window must be > 0")
        if self.config.margin < 0:
            raise ValueError("margin must be >= 0")

        self.use_kv_cache = self.config.strategy != "no_kv_cache"

        self.block_stride = max(1, self.config.context_window - self.config.margin)

        if self._inference_backend == "huggingface":
            if "<EOF>" not in self.tokenizer.all_special_tokens:
                self.tokenizer.add_special_tokens({"additional_special_tokens": ["<EOF>"]})
                self.model.resize_token_embeddings(len(self.tokenizer))
            self.eof_token_id = self.tokenizer.convert_tokens_to_ids("<EOF>")
        else:
            self.eof_token_id = self._resolve_existing_eof_token_id()

        self.dec_prec = max(50, int(math.ceil(self.config.precision * math.log10(2))) + 10)

        self._batch_invariant_enabled = False
        self._log_softmax_fn = torch.log_softmax
        self._batch_invariant_ctx = nullcontext

        if self.model is not None:
            try:
                self.model.config._attn_implementation = "eager"
            except Exception:
                pass

        self._configure_determinism_runtime()
        if self._inference_backend == "vllm":
            model_id = self.config.model_id
            if not model_id:
                raise ValueError("vLLM backend requires config.model_id")
            self._vllm_backend = _VLLMLogitsBackend(
                model_id=str(model_id),
                tokenizer=self.tokenizer,
                device=self.device,
                trust_remote_code=bool(self.config.trust_remote_code),
                revision=self.config.revision,
                torch_dtype=str(self.config.torch_dtype),
                tensor_parallel_size=int(self.config.vllm_tensor_parallel_size),
                gpu_memory_utilization=float(self.config.vllm_gpu_memory_utilization),
                max_logprobs=self.config.vllm_max_logprobs,
            )

    def _resolve_inference_backend(self) -> str:
        backend = str(self.config.inference_backend or "auto").strip().lower()
        if backend in {"hf", "transformers", "huggingface"}:
            return "huggingface"
        if backend in {"vllm"}:
            return "vllm"
        if backend not in {"", "auto"}:
            raise ValueError("Unsupported inference_backend. Expected one of: auto, huggingface, vllm")

        if self._determinism_mode is None:
            return "huggingface"
        if self.device.type != "cuda":
            return "huggingface"
        if not _triton_is_available():
            return "huggingface"
        return "vllm"

    def _resolve_existing_eof_token_id(self) -> int:
        for token_attr in ("eos_token_id", "eod_id", "im_end_id"):
            token_id = getattr(self.tokenizer, token_attr, None)
            if isinstance(token_id, int) and token_id >= 0:
                return int(token_id)

        token_id = self.tokenizer.convert_tokens_to_ids("<EOF>")
        if isinstance(token_id, int) and token_id >= 0:
            return int(token_id)

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if isinstance(pad_id, int) and pad_id >= 0:
            return int(pad_id)
        return 0

    def _diagnostics_enabled(self) -> bool:
        return bool(self.config.diagnostics_csv_prefix)

    @property
    def inference_backend(self) -> str:
        return self._inference_backend

    def _diagnostics_csv_path(self, phase: str) -> Path:
        base = Path(str(self.config.diagnostics_csv_prefix))
        if base.suffix.lower() == ".csv":
            return base.with_name(f"{base.stem}_{phase}.csv")
        return Path(str(base) + f"_{phase}.csv")

    @staticmethod
    def _diagnostics_fieldnames():
        return [
            "phase",
            "step_idx",
            "token_id",
            "token_text",
            "coder_D_before",
            "selected_interval_contains_D",
            "raw_cum_prob",
            "effective_cum_prob",
            "raw_lower_prob",
            "raw_upper_prob",
            "effective_lower_prob",
            "effective_upper_prob",
            "prev_token_id",
            "prev_token_text",
            "raw_prev_lower_prob",
            "raw_prev_upper_prob",
            "effective_prev_lower_prob",
            "effective_prev_upper_prob",
            "raw_prev_interval_low",
            "raw_prev_interval_high",
            "effective_prev_interval_low",
            "effective_prev_interval_high",
            "next_token_id",
            "next_token_text",
            "raw_next_lower_prob",
            "raw_next_upper_prob",
            "effective_next_lower_prob",
            "effective_next_upper_prob",
            "raw_next_interval_low",
            "raw_next_interval_high",
            "effective_next_interval_low",
            "effective_next_interval_high",
            "raw_gap_prev_to_current",
            "raw_gap_current_to_next",
            "effective_gap_prev_to_current",
            "effective_gap_current_to_next",
            "raw_interval_gap_prev_to_current",
            "raw_interval_gap_current_to_next",
            "effective_interval_gap_prev_to_current",
            "effective_interval_gap_current_to_next",
            "raw_interval_low",
            "raw_interval_high",
            "effective_interval_low",
            "effective_interval_high",
            "coder_L_before",
            "coder_R_before",
            "rank_raw",
            "rank_effective",
            "top1_raw_token_id",
            "top1_raw_prob",
            "top1_effective_token_id",
            "top1_effective_prob",
            "symbol_count",
            "counts_total",
        ]

    def _open_diagnostics_writer(self, phase: str):
        if not self._diagnostics_enabled():
            return None, None

        path = self._diagnostics_csv_path(phase)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=self._diagnostics_fieldnames())
        writer.writeheader()
        return handle, writer

    @staticmethod
    def _token_text_for_diag(tokenizer, token_id: int) -> str:
        try:
            return tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")
        except Exception:
            return ""

    @staticmethod
    def _rank_from_probs(probs: np.ndarray, token_id: int) -> int:
        token_prob = float(probs[int(token_id)])
        return int(np.count_nonzero(probs > token_prob) + 1)

    @staticmethod
    def _interval_bounds_from_probs(
        probs: np.ndarray,
        token_id: int,
        coder_L: int,
        coder_R: int,
    ):
        token_id = int(token_id)
        lower_prob = float(np.sum(probs[:token_id]))
        token_prob = float(probs[token_id])
        upper_prob = lower_prob + token_prob

        low = int(coder_L + math.floor(coder_R * lower_prob))
        high = int(coder_L + math.floor(coder_R * upper_prob) - 1)
        return lower_prob, upper_prob, low, high

    @staticmethod
    def _interval_bounds_from_counts(
        counts: np.ndarray,
        token_id: int,
        coder_L: int,
        coder_R: int,
    ):
        token_id = int(token_id)
        total = int(np.sum(counts))
        lower_count = int(np.sum(counts[:token_id]))
        symbol_count = int(counts[token_id])
        upper_count = lower_count + symbol_count

        lower_prob = (lower_count / total) if total > 0 else 0.0
        upper_prob = (upper_count / total) if total > 0 else 0.0

        low = int(coder_L + (coder_R * lower_count) // total) if total > 0 else coder_L
        high = int(coder_L + (coder_R * upper_count) // total - 1) if total > 0 else (coder_L - 1)
        return lower_prob, upper_prob, low, high, symbol_count, total

    def _write_token_diagnostic(
        self,
        writer,
        phase: str,
        step_idx: int,
        token_id: int,
        raw_probs: np.ndarray,
        effective_probs: np.ndarray,
        counts,
        coder_L_before: int,
        coder_R_before: int,
        coder_D_before: Optional[int] = None,
    ):
        if writer is None:
            return

        token_id = int(token_id)
        counts_np = np.asarray(counts, dtype=np.int64)

        raw_lower, raw_upper, raw_low, raw_high = self._interval_bounds_from_probs(
            raw_probs,
            token_id,
            coder_L_before,
            coder_R_before,
        )
        eff_lower, eff_upper, eff_low, eff_high, symbol_count, counts_total = self._interval_bounds_from_counts(
            counts_np,
            token_id,
            coder_L_before,
            coder_R_before,
        )

        prev_token_id = token_id - 1 if token_id > 0 else None
        next_token_id = token_id + 1 if token_id < (len(raw_probs) - 1) else None

        def _neighbor_bounds(neighbor_id: Optional[int]):
            if neighbor_id is None:
                return None
            n_raw_lower, n_raw_upper, n_raw_low, n_raw_high = self._interval_bounds_from_probs(
                raw_probs,
                neighbor_id,
                coder_L_before,
                coder_R_before,
            )
            n_eff_lower, n_eff_upper, n_eff_low, n_eff_high, _, _ = self._interval_bounds_from_counts(
                counts_np,
                neighbor_id,
                coder_L_before,
                coder_R_before,
            )
            return {
                "token_id": neighbor_id,
                "token_text": self._token_text_for_diag(self.tokenizer, neighbor_id),
                "raw_lower": n_raw_lower,
                "raw_upper": n_raw_upper,
                "eff_lower": n_eff_lower,
                "eff_upper": n_eff_upper,
                "raw_low": n_raw_low,
                "raw_high": n_raw_high,
                "eff_low": n_eff_low,
                "eff_high": n_eff_high,
            }

        prev_bounds = _neighbor_bounds(prev_token_id)
        next_bounds = _neighbor_bounds(next_token_id)

        raw_gap_prev_to_current = ""
        raw_gap_current_to_next = ""
        eff_gap_prev_to_current = ""
        eff_gap_current_to_next = ""
        raw_interval_gap_prev_to_current = ""
        raw_interval_gap_current_to_next = ""
        eff_interval_gap_prev_to_current = ""
        eff_interval_gap_current_to_next = ""

        if prev_bounds is not None:
            raw_gap_prev_to_current = raw_lower - prev_bounds["raw_upper"]
            eff_gap_prev_to_current = eff_lower - prev_bounds["eff_upper"]
            raw_interval_gap_prev_to_current = raw_low - (prev_bounds["raw_high"] + 1)
            eff_interval_gap_prev_to_current = eff_low - (prev_bounds["eff_high"] + 1)

        if next_bounds is not None:
            raw_gap_current_to_next = next_bounds["raw_lower"] - raw_upper
            eff_gap_current_to_next = next_bounds["eff_lower"] - eff_upper
            raw_interval_gap_current_to_next = next_bounds["raw_low"] - (raw_high + 1)
            eff_interval_gap_current_to_next = next_bounds["eff_low"] - (eff_high + 1)

        contains_d = ""
        if coder_D_before is not None:
            contains_d = int(eff_low <= int(coder_D_before) <= eff_high)

        top1_raw = int(np.argmax(raw_probs))
        top1_eff = int(np.argmax(effective_probs))

        writer.writerow(
            {
                "phase": phase,
                "step_idx": int(step_idx),
                "token_id": token_id,
                "token_text": self._token_text_for_diag(self.tokenizer, token_id),
                "coder_D_before": "" if coder_D_before is None else int(coder_D_before),
                "selected_interval_contains_D": contains_d,
                # CDF value at selected token (inclusive upper bound).
                "raw_cum_prob": raw_upper,
                "effective_cum_prob": eff_upper,
                "raw_lower_prob": raw_lower,
                "raw_upper_prob": raw_upper,
                "effective_lower_prob": eff_lower,
                "effective_upper_prob": eff_upper,
                "prev_token_id": "" if prev_bounds is None else int(prev_bounds["token_id"]),
                "prev_token_text": "" if prev_bounds is None else prev_bounds["token_text"],
                "raw_prev_lower_prob": "" if prev_bounds is None else prev_bounds["raw_lower"],
                "raw_prev_upper_prob": "" if prev_bounds is None else prev_bounds["raw_upper"],
                "effective_prev_lower_prob": "" if prev_bounds is None else prev_bounds["eff_lower"],
                "effective_prev_upper_prob": "" if prev_bounds is None else prev_bounds["eff_upper"],
                "raw_prev_interval_low": "" if prev_bounds is None else prev_bounds["raw_low"],
                "raw_prev_interval_high": "" if prev_bounds is None else prev_bounds["raw_high"],
                "effective_prev_interval_low": "" if prev_bounds is None else prev_bounds["eff_low"],
                "effective_prev_interval_high": "" if prev_bounds is None else prev_bounds["eff_high"],
                "next_token_id": "" if next_bounds is None else int(next_bounds["token_id"]),
                "next_token_text": "" if next_bounds is None else next_bounds["token_text"],
                "raw_next_lower_prob": "" if next_bounds is None else next_bounds["raw_lower"],
                "raw_next_upper_prob": "" if next_bounds is None else next_bounds["raw_upper"],
                "effective_next_lower_prob": "" if next_bounds is None else next_bounds["eff_lower"],
                "effective_next_upper_prob": "" if next_bounds is None else next_bounds["eff_upper"],
                "raw_next_interval_low": "" if next_bounds is None else next_bounds["raw_low"],
                "raw_next_interval_high": "" if next_bounds is None else next_bounds["raw_high"],
                "effective_next_interval_low": "" if next_bounds is None else next_bounds["eff_low"],
                "effective_next_interval_high": "" if next_bounds is None else next_bounds["eff_high"],
                "raw_gap_prev_to_current": raw_gap_prev_to_current,
                "raw_gap_current_to_next": raw_gap_current_to_next,
                "effective_gap_prev_to_current": eff_gap_prev_to_current,
                "effective_gap_current_to_next": eff_gap_current_to_next,
                "raw_interval_gap_prev_to_current": raw_interval_gap_prev_to_current,
                "raw_interval_gap_current_to_next": raw_interval_gap_current_to_next,
                "effective_interval_gap_prev_to_current": eff_interval_gap_prev_to_current,
                "effective_interval_gap_current_to_next": eff_interval_gap_current_to_next,
                "raw_interval_low": raw_low,
                "raw_interval_high": raw_high,
                "effective_interval_low": eff_low,
                "effective_interval_high": eff_high,
                "coder_L_before": int(coder_L_before),
                "coder_R_before": int(coder_R_before),
                "rank_raw": self._rank_from_probs(raw_probs, token_id),
                "rank_effective": self._rank_from_probs(effective_probs, token_id),
                "top1_raw_token_id": top1_raw,
                "top1_raw_prob": float(raw_probs[top1_raw]),
                "top1_effective_token_id": top1_eff,
                "top1_effective_prob": float(effective_probs[top1_eff]),
                "symbol_count": symbol_count,
                "counts_total": counts_total,
            }
        )

    def _configure_determinism_runtime(self):
        self._batch_invariant_enabled = False
        self._batch_invariant_ctx = lambda _enabled=True: nullcontext()
        self._log_softmax_fn = torch.log_softmax

        if self._determinism_mode is None:
            return

        # CPU path: mimic batch-invariant fixed-order reductions for probability extraction.
        if self.device.type == "cpu":
            self._batch_invariant_enabled = True
            self._batch_invariant_ctx = lambda _enabled=True: nullcontext()
            self._log_softmax_fn = _cpu_batch_invariant_log_softmax
            return

        if self._determinism_mode == "tbik":
            if self.device.type != "cuda":
                self._batch_invariant_enabled = True
                self._batch_invariant_ctx = lambda _enabled=True: nullcontext()
                self._log_softmax_fn = _cpu_batch_invariant_log_softmax
                return
            _apply_tbik_patches()

        prefer_repo_backend = self._determinism_mode == "tbik"
        set_batch_invariant_mode, log_softmax = _import_batch_invariant_backend(
            prefer_repo_backend=prefer_repo_backend
        )
        if set_batch_invariant_mode is None or log_softmax is None:
            raise RuntimeError(
                f"determinism_mode='{self._determinism_mode}' requires batch invariant ops, but no backend could be imported"
            )

        try:
            sample = torch.zeros((1, 4), dtype=torch.float32, device=self.device)
            with set_batch_invariant_mode(True):
                _ = log_softmax(sample, dim=-1)
            self._batch_invariant_enabled = True
            self._batch_invariant_ctx = set_batch_invariant_mode
            self._log_softmax_fn = log_softmax
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize determinism_mode='{self._determinism_mode}' backend: {exc}"
            ) from exc

    def _invariant_context(self):
        if self._batch_invariant_enabled:
            return self._batch_invariant_ctx(True)
        return nullcontext()

    @staticmethod
    def _position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return position_ids

    def _logits_for_prefix(self, prefix_ids):
        if self._vllm_backend is not None:
            return self._vllm_backend.next_logits(prefix_ids)

        if len(prefix_ids) == 0:
            bos = self.tokenizer.bos_token_id
            if bos is None:
                bos = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids = torch.tensor([[bos]], dtype=torch.long, device=self.device)
        else:
            input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)

        attention_mask = torch.ones_like(input_ids)
        position_ids = self._position_ids_from_mask(attention_mask)

        with torch.no_grad():
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=self.use_kv_cache,
            )
        return out.logits[0, -1, :]

    def _init_cache_state(self):
        return {
            "past_key_values": None,
            "next_logits": None,
            "cached_token_count": 0,
        }

    def _ensure_dynamic_cache(self, past_kv):
        if past_kv is None:
            return None
        if hasattr(past_kv, "key_cache"):
            return past_kv
        if isinstance(past_kv, tuple) and HAS_DYNAMIC_CACHE:
            try:
                return DynamicCache.from_legacy_cache(past_kv)
            except Exception:
                return past_kv
        return past_kv

    def _hard_reset_cache_and_warmup(self, full_sequence, current_index, warmup_length):
        if self._vllm_backend is not None:
            return self._init_cache_state()

        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()

        start = max(0, current_index - warmup_length)
        warmup_tokens = full_sequence[start:current_index]
        if len(warmup_tokens) == 0:
            return self._init_cache_state()

        input_ids = torch.tensor([warmup_tokens], dtype=torch.long).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)

        return {
            "past_key_values": outputs.past_key_values,
            "next_logits": outputs.logits[0, -1, :],
            "cached_token_count": len(warmup_tokens),
        }

    def _truncate_cache_rolling(self, cache_state):
        past_kv = cache_state["past_key_values"]
        limit = self.config.context_window
        sink = 4

        if past_kv is None:
            return cache_state

        if hasattr(past_kv, "key_cache"):
            current_len = past_kv.key_cache[0].size(2)
            if current_len <= limit:
                return cache_state

            keep_recent = max(1, limit - sink)
            new_keys = []
            new_values = []
            for key, value in zip(past_kv.key_cache, past_kv.value_cache):
                key_sink = key[:, :, :sink, :]
                key_recent = key[:, :, -keep_recent:, :]
                value_sink = value[:, :, :sink, :]
                value_recent = value[:, :, -keep_recent:, :]
                new_keys.append(torch.cat([key_sink, key_recent], dim=2))
                new_values.append(torch.cat([value_sink, value_recent], dim=2))

            past_kv.key_cache = new_keys
            past_kv.value_cache = new_values
            if hasattr(past_kv, "_seen_tokens"):
                past_kv._seen_tokens = limit

            cache_state["past_key_values"] = past_kv
            cache_state["cached_token_count"] = limit
            return cache_state

        if isinstance(past_kv, tuple):
            current_len = past_kv[0][0].size(2)
            if current_len <= limit:
                return cache_state

            keep_recent = max(1, limit - sink)
            new_past = []
            for key, value in past_kv:
                new_key = torch.cat([key[:, :, :sink, :], key[:, :, -keep_recent:, :]], dim=2)
                new_value = torch.cat([value[:, :, :sink, :], value[:, :, -keep_recent:, :]], dim=2)
                new_past.append((new_key, new_value))

            cache_state["past_key_values"] = tuple(new_past)
            cache_state["cached_token_count"] = limit
            return cache_state

        return cache_state

    def _get_logits(self, current_idx, full_token_sequence, cache_state):
        if self._vllm_backend is not None:
            return self._vllm_backend.next_logits(full_token_sequence[:current_idx])

        if not self.use_kv_cache:
            start = max(0, current_idx - self.config.context_window)
            context = full_token_sequence[start:current_idx]
            if len(context) == 0:
                bos = self.tokenizer.bos_token_id
                if bos is None:
                    bos = self.tokenizer.pad_token_id
                if bos is None:
                    return torch.log(torch.ones(self.tokenizer.vocab_size).to(self.device))
                context = [bos]

            input_ids = torch.tensor([context], dtype=torch.long).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, use_cache=False)
            return outputs.logits[0, -1, :]

        if cache_state["next_logits"] is not None:
            return cache_state["next_logits"]

        if current_idx == 0:
            bos = self.tokenizer.bos_token_id
            if bos is None:
                bos = self.tokenizer.pad_token_id
            if bos is not None:
                input_ids = torch.tensor([[bos]], dtype=torch.long).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, use_cache=True)
                return outputs.logits[0, -1, :]

        return torch.log(torch.ones(self.tokenizer.vocab_size).to(self.device))

    def _advance_state(self, just_encoded_token_id, cache_state):
        if self._vllm_backend is not None:
            return cache_state

        if not self.use_kv_cache:
            return cache_state

        past_kv = self._ensure_dynamic_cache(cache_state["past_key_values"])
        input_ids = torch.tensor([[just_encoded_token_id]], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, past_key_values=past_kv, use_cache=True)

        cache_state["past_key_values"] = outputs.past_key_values
        cache_state["next_logits"] = outputs.logits[0, -1, :]
        cache_state["cached_token_count"] += 1
        return cache_state

    def _raw_and_effective_probs(self, logits: torch.Tensor):
        logits_2d = logits.view(1, -1).detach().to(dtype=torch.float32)
        raw_t = torch.exp(self._log_softmax_fn(logits_2d, dim=-1))[0]
        raw_probs = raw_t.detach().cpu().numpy().astype(np.float64, copy=False)

        raw_sum = raw_probs.sum()
        if raw_sum <= 0:
            raw_probs = np.full_like(raw_probs, 1.0 / len(raw_probs), dtype=np.float64)
        else:
            raw_probs = raw_probs / raw_sum

        if not self.config.quant:
            return raw_probs, raw_probs.copy()

        logits_np = (
            logits.detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
            .reshape(-1)
            .astype(np.float64, copy=False)
        )

        if self.config.logit_round_decimals >= 0:
            scale = 10 ** int(self.config.logit_round_decimals)
            logits_np = np.rint(logits_np * scale) / float(scale)

        max_logit = float(np.max(logits_np))
        exp_shifted = np.exp(logits_np - max_logit)
        eff_probs = exp_shifted / np.sum(exp_shifted)

        if self.config.prob_round_decimals >= 0:
            scale = 10 ** int(self.config.prob_round_decimals)
            fixed = np.rint(eff_probs * scale).astype(np.int64)
            fixed = np.clip(fixed, 0, None)
            fixed_sum = int(fixed.sum())

            if fixed_sum <= 0:
                fixed = np.zeros_like(fixed)
                fixed[int(np.argmax(eff_probs))] = 1
                fixed_sum = 1

            eff_probs = fixed.astype(np.float64) / float(fixed_sum)

        eff_probs = np.clip(eff_probs, 0.0, None)
        eff_sum = float(np.sum(eff_probs))
        if eff_sum <= 0.0:
            eff_probs = np.full_like(eff_probs, 1.0 / len(eff_probs), dtype=np.float64)
        else:
            eff_probs = eff_probs / eff_sum

        return raw_probs, eff_probs

    def _probs(self, logits: torch.Tensor) -> np.ndarray:
        _, effective_probs = self._raw_and_effective_probs(logits)
        return effective_probs

    def _counts_from_probs(self, probs: np.ndarray):
        if self.config.use_legacy_counts:
            return probs_to_counts_legacy(probs, self.config.slots, self.dec_prec)
        return probs_to_counts(probs, self.config.slots, self.dec_prec)

    @staticmethod
    def _write_csv_rows(path: str, rows: List[Dict[str, Any]]):
        if not rows:
            return
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _start_memory_monitor(self, enabled: bool, sample_interval: float):
        if not enabled:
            return None

        try:
            import psutil
        except Exception:
            return None

        stop_event = threading.Event()
        rows: List[Dict[str, float]] = []
        process = psutil.Process(os.getpid())
        start_time = time.perf_counter()
        has_cuda = torch.cuda.is_available()
        has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

        mps_alloc_fn = None
        if has_mps and hasattr(torch, "mps"):
            mps_alloc_fn = getattr(torch.mps, "current_allocated_memory", None)

        def _monitor():
            while not stop_event.is_set():
                try:
                    mem_info = process.memory_info()
                    cuda_allocated_mb = ""
                    cuda_reserved_mb = ""
                    mps_allocated_mb = ""
                    vram_mb = ""

                    if has_cuda:
                        try:
                            cuda_allocated_mb = float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
                            cuda_reserved_mb = float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)
                            vram_mb = cuda_allocated_mb
                        except Exception:
                            pass

                    if has_mps and mps_alloc_fn is not None:
                        try:
                            mps_allocated_mb = float(mps_alloc_fn()) / (1024.0 * 1024.0)
                            if vram_mb == "":
                                vram_mb = mps_allocated_mb
                        except Exception:
                            pass

                    rows.append(
                        {
                            "time": time.perf_counter() - start_time,
                            "rss_mb": float(mem_info.rss) / (1024.0 * 1024.0),
                            "vms_mb": float(mem_info.vms) / (1024.0 * 1024.0),
                            "vram_mb": vram_mb,
                            "cuda_allocated_mb": cuda_allocated_mb,
                            "cuda_reserved_mb": cuda_reserved_mb,
                            "mps_allocated_mb": mps_allocated_mb,
                        }
                    )
                except Exception:
                    break
                time.sleep(max(0.01, float(sample_interval)))

        thread = threading.Thread(target=_monitor, daemon=True)
        thread.start()
        return {"stop_event": stop_event, "thread": thread, "rows": rows}

    def _stop_memory_monitor(self, monitor_state, speed_rows: List[Dict[str, Any]]):
        if monitor_state is None:
            return []

        monitor_state["stop_event"].set()
        monitor_state["thread"].join()
        rows = monitor_state["rows"]
        if not rows:
            return []

        if speed_rows:
            elapsed = np.cumsum([float(row["time"]) for row in speed_rows])
            pos = np.asarray([float(row["pos"]) for row in speed_rows], dtype=np.float64)
            for row in rows:
                row["aligned_pos"] = float(np.interp(float(row["time"]), elapsed, pos))
        else:
            for row in rows:
                row["aligned_pos"] = ""

        return rows

    def _maybe_write_divergence_rows(
        self,
        demo_csv_path: str,
        decoded_ids: Sequence[int],
        reference_token_ids: Optional[Sequence[int]],
        divergence_window: int,
    ):
        if reference_token_ids is None:
            return

        reference_ids = [int(x) for x in reference_token_ids]
        decoded = [int(x) for x in decoded_ids]
        compare_len = min(len(decoded), len(reference_ids))

        first_div_idx = None
        for idx in range(compare_len):
            if decoded[idx] != reference_ids[idx]:
                first_div_idx = idx
                break

        if first_div_idx is None and len(decoded) != len(reference_ids):
            first_div_idx = compare_len

        rows: List[Dict[str, Any]] = []
        if first_div_idx is None:
            rows.append(
                {
                    "segment": "summary",
                    "step_idx": "",
                    "reference_token_id": "",
                    "reference_token_text": "",
                    "decoded_token_id": "",
                    "decoded_token_text": "",
                    "match": 1,
                    "note": "No divergence detected in compared token sequence.",
                }
            )
        else:
            start_match = max(0, first_div_idx - int(max(1, divergence_window)))
            for idx in range(start_match, first_div_idx):
                ref_id = reference_ids[idx]
                dec_id = decoded[idx]
                rows.append(
                    {
                        "segment": "last_matching",
                        "step_idx": idx,
                        "reference_token_id": ref_id,
                        "reference_token_text": self._token_text_for_diag(self.tokenizer, ref_id),
                        "decoded_token_id": dec_id,
                        "decoded_token_text": self._token_text_for_diag(self.tokenizer, dec_id),
                        "match": int(ref_id == dec_id),
                        "note": "",
                    }
                )

            max_len = max(len(decoded), len(reference_ids))
            end_div = min(max_len, first_div_idx + int(max(1, divergence_window)))
            for idx in range(first_div_idx, end_div):
                ref_id = reference_ids[idx] if idx < len(reference_ids) else None
                dec_id = decoded[idx] if idx < len(decoded) else None
                rows.append(
                    {
                        "segment": "first_diverged",
                        "step_idx": idx,
                        "reference_token_id": "" if ref_id is None else ref_id,
                        "reference_token_text": ""
                        if ref_id is None
                        else self._token_text_for_diag(self.tokenizer, ref_id),
                        "decoded_token_id": "" if dec_id is None else dec_id,
                        "decoded_token_text": ""
                        if dec_id is None
                        else self._token_text_for_diag(self.tokenizer, dec_id),
                        "match": int(ref_id is not None and dec_id is not None and ref_id == dec_id),
                        "note": "",
                    }
                )

        divergence_path = Path(demo_csv_path)
        divergence_path = divergence_path.with_name(f"{divergence_path.stem}_divergence.csv")
        self._write_csv_rows(str(divergence_path), rows)

    def encode(
        self,
        text: str,
        safe_mode: bool = False,
        return_token_count: bool = False,
        show_progress: bool = True,
        demo: bool = False,
        demo_csv_path: str = "compression_stats.csv",
        speed_demo: bool = False,
        speed_csv_path: str = "speed_encode.csv",
        memory_demo: bool = False,
        memory_csv_path: str = "memory_encode.csv",
        memory_sample_interval: float = 0.05,
    ):
        token_ids = self.tokenizer.encode(text)
        if not safe_mode:
            token_ids = token_ids + [self.eof_token_id]

        writer = BitWriter()
        enc = Encoder(Coder(b=self.config.precision), writer)
        iterator = enumerate(token_ids)
        if show_progress:
            iterator = tqdm(iterator, total=len(token_ids), desc="Deterministic Encode")

        cache_state = self._init_cache_state()
        diag_handle, diag_writer = self._open_diagnostics_writer("encode")
        demo_rows: List[Dict[str, Any]] = []
        speed_rows: List[Dict[str, Any]] = []
        monitor_state = self._start_memory_monitor(memory_demo, memory_sample_interval)

        try:
            with self._invariant_context():
                if self.use_kv_cache:
                    cache_state = self._hard_reset_cache_and_warmup([], 0, 0)
                    cache_state["next_logits"] = self._get_logits(0, token_ids, cache_state)

                for idx, token_id in iterator:
                    iter_start = time.perf_counter()
                    if self.config.strategy == "block" and idx > 0 and (idx % self.block_stride == 0):
                        cache_state = self._hard_reset_cache_and_warmup(
                            token_ids,
                            current_index=idx,
                            warmup_length=self.config.margin,
                        )

                    logits = self._get_logits(idx, token_ids, cache_state)
                    raw_probs, probs = self._raw_and_effective_probs(logits)
                    counts = self._counts_from_probs(probs)

                    if demo:
                        p_raw = float(raw_probs[int(token_id)])
                        p_eff = float(probs[int(token_id)])
                        safe_raw = max(p_raw, 1e-12)
                        demo_rows.append(
                            {
                                "pos": int(idx),
                                "token_id": int(token_id),
                                "token": self._token_text_for_diag(self.tokenizer, int(token_id)),
                                "prob_raw": p_raw,
                                "prob_effective": p_eff,
                                "perplexity_raw": 1.0 / safe_raw,
                                "surprisal_bits_raw": -math.log2(safe_raw),
                            }
                        )

                    coder_L_before = int(enc.coder.L)
                    coder_R_before = int(enc.coder.R)
                    self._write_token_diagnostic(
                        diag_writer,
                        phase="encode",
                        step_idx=idx,
                        token_id=int(token_id),
                        raw_probs=raw_probs,
                        effective_probs=probs,
                        counts=counts,
                        coder_L_before=coder_L_before,
                        coder_R_before=coder_R_before,
                        coder_D_before=None,
                    )

                    enc.encode_symbol(token_id, counts_to_cum_desc(counts))

                    if idx < len(token_ids) - 1:
                        cache_state = self._advance_state(token_id, cache_state)
                        if (
                            self.config.strategy == "rolling"
                            and cache_state["cached_token_count"]
                            > (self.config.context_window + self.config.margin)
                        ):
                            cache_state = self._truncate_cache_rolling(cache_state)

                    if speed_demo or memory_demo:
                        speed_rows.append({"pos": int(idx), "time": time.perf_counter() - iter_start})
        finally:
            if diag_handle is not None:
                diag_handle.close()
            if memory_demo:
                memory_rows = self._stop_memory_monitor(monitor_state, speed_rows)
                self._write_csv_rows(memory_csv_path, memory_rows)

        enc.finish()
        writer.flush()
        encoded_bytes = writer.getvalue()

        if speed_demo:
            self._write_csv_rows(speed_csv_path, speed_rows)
        if demo:
            self._write_csv_rows(demo_csv_path, demo_rows)

        if safe_mode or return_token_count:
            return encoded_bytes, len(token_ids)
        return encoded_bytes

    def decode(
        self,
        encoded_bytes: bytes,
        max_decode_tokens: Optional[int] = None,
        safe_mode: bool = False,
        expected_num_tokens: Optional[int] = None,
        show_progress: bool = True,
        demo: bool = False,
        demo_csv_path: str = "demo_decode.csv",
        speed_demo: bool = False,
        speed_csv_path: str = "speed_decode.csv",
        memory_demo: bool = False,
        memory_csv_path: str = "memory_decode.csv",
        memory_sample_interval: float = 0.05,
        reference_token_ids: Optional[Sequence[int]] = None,
        divergence_window: int = 5,
    ) -> str:
        dec = Decoder(Coder(b=self.config.precision), BitReader(encoded_bytes))
        decoded_ids = []
        cache_state = self._init_cache_state()
        diag_handle, diag_writer = self._open_diagnostics_writer("decode")
        demo_rows: List[Dict[str, Any]] = []
        speed_rows: List[Dict[str, Any]] = []
        monitor_state = self._start_memory_monitor(memory_demo, memory_sample_interval)
        decoded_text = ""

        if safe_mode and expected_num_tokens is None:
            raise ValueError("safe_mode=True requires expected_num_tokens to be provided.")

        try:
            with self._invariant_context():
                if self.use_kv_cache:
                    cache_state = self._hard_reset_cache_and_warmup([], 0, 0)
                    cache_state["next_logits"] = self._get_logits(0, [], cache_state)

                if safe_mode:
                    target_tokens = int(expected_num_tokens)
                    iterator = range(target_tokens)
                    if show_progress:
                        iterator = tqdm(iterator, total=target_tokens, desc="Deterministic Decode")

                    for _ in iterator:
                        iter_start = time.perf_counter()
                        idx = len(decoded_ids)
                        if max_decode_tokens is not None and idx >= max_decode_tokens:
                            raise RuntimeError(
                                f"Decoding exceeded max_decode_tokens={max_decode_tokens} before target token count."
                            )

                        if self.config.strategy == "block" and idx > 0 and (idx % self.block_stride == 0):
                            cache_state = self._hard_reset_cache_and_warmup(
                                decoded_ids,
                                current_index=idx,
                                warmup_length=self.config.margin,
                            )

                        logits = self._get_logits(idx, decoded_ids, cache_state)
                        raw_probs, probs = self._raw_and_effective_probs(logits)
                        counts = self._counts_from_probs(probs)

                        coder_L_before = int(dec.coder.L)
                        coder_R_before = int(dec.coder.R)
                        coder_D_before = int(dec.coder.D)
                        token_id = dec.decode_symbol(counts_to_cum_desc(counts))

                        self._write_token_diagnostic(
                            diag_writer,
                            phase="decode",
                            step_idx=idx,
                            token_id=int(token_id),
                            raw_probs=raw_probs,
                            effective_probs=probs,
                            counts=counts,
                            coder_L_before=coder_L_before,
                            coder_R_before=coder_R_before,
                            coder_D_before=coder_D_before,
                        )

                        if demo:
                            p_raw = float(raw_probs[int(token_id)])
                            p_eff = float(probs[int(token_id)])
                            safe_raw = max(p_raw, 1e-12)
                            row: Dict[str, Any] = {
                                "pos": int(idx),
                                "decoded_token_id": int(token_id),
                                "decoded_token": self._token_text_for_diag(self.tokenizer, int(token_id)),
                                "prob_raw": p_raw,
                                "prob_effective": p_eff,
                                "perplexity_raw": 1.0 / safe_raw,
                                "surprisal_bits_raw": -math.log2(safe_raw),
                            }
                            if reference_token_ids is not None and idx < len(reference_token_ids):
                                ref_id = int(reference_token_ids[idx])
                                row["reference_token_id"] = ref_id
                                row["reference_token"] = self._token_text_for_diag(self.tokenizer, ref_id)
                                row["match"] = int(ref_id == int(token_id))
                            demo_rows.append(row)

                        decoded_ids.append(token_id)

                        cache_state = self._advance_state(token_id, cache_state)
                        if (
                            self.config.strategy == "rolling"
                            and cache_state["cached_token_count"]
                            > (self.config.context_window + self.config.margin)
                        ):
                            cache_state = self._truncate_cache_rolling(cache_state)

                        if speed_demo or memory_demo:
                            speed_rows.append({"pos": int(idx), "time": time.perf_counter() - iter_start})

                    decoded_text = self.tokenizer.decode(decoded_ids, skip_special_tokens=True)
                else:
                    token_id = None
                    decode_iterator = None
                    if show_progress:
                        decode_iterator = tqdm(desc="Deterministic Decode", unit="tok")
                    while token_id != self.eof_token_id:
                        iter_start = time.perf_counter()
                        idx = len(decoded_ids)
                        if max_decode_tokens is not None and idx >= max_decode_tokens:
                            raise RuntimeError(
                                f"Decoding exceeded max_decode_tokens={max_decode_tokens} before EOF."
                            )

                        if self.config.strategy == "block" and idx > 0 and (idx % self.block_stride == 0):
                            cache_state = self._hard_reset_cache_and_warmup(
                                decoded_ids,
                                current_index=idx,
                                warmup_length=self.config.margin,
                            )

                        logits = self._get_logits(idx, decoded_ids, cache_state)
                        raw_probs, probs = self._raw_and_effective_probs(logits)
                        counts = self._counts_from_probs(probs)

                        coder_L_before = int(dec.coder.L)
                        coder_R_before = int(dec.coder.R)
                        coder_D_before = int(dec.coder.D)
                        token_id = dec.decode_symbol(counts_to_cum_desc(counts))

                        self._write_token_diagnostic(
                            diag_writer,
                            phase="decode",
                            step_idx=idx,
                            token_id=int(token_id),
                            raw_probs=raw_probs,
                            effective_probs=probs,
                            counts=counts,
                            coder_L_before=coder_L_before,
                            coder_R_before=coder_R_before,
                            coder_D_before=coder_D_before,
                        )

                        if demo:
                            p_raw = float(raw_probs[int(token_id)])
                            p_eff = float(probs[int(token_id)])
                            safe_raw = max(p_raw, 1e-12)
                            row = {
                                "pos": int(idx),
                                "decoded_token_id": int(token_id),
                                "decoded_token": self._token_text_for_diag(self.tokenizer, int(token_id)),
                                "prob_raw": p_raw,
                                "prob_effective": p_eff,
                                "perplexity_raw": 1.0 / safe_raw,
                                "surprisal_bits_raw": -math.log2(safe_raw),
                            }
                            if reference_token_ids is not None and idx < len(reference_token_ids):
                                ref_id = int(reference_token_ids[idx])
                                row["reference_token_id"] = ref_id
                                row["reference_token"] = self._token_text_for_diag(self.tokenizer, ref_id)
                                row["match"] = int(ref_id == int(token_id))
                            demo_rows.append(row)

                        decoded_ids.append(token_id)

                        if token_id != self.eof_token_id:
                            cache_state = self._advance_state(token_id, cache_state)
                            if (
                                self.config.strategy == "rolling"
                                and cache_state["cached_token_count"]
                                > (self.config.context_window + self.config.margin)
                            ):
                                cache_state = self._truncate_cache_rolling(cache_state)
                        if decode_iterator is not None:
                            decode_iterator.update(1)

                        if speed_demo or memory_demo:
                            speed_rows.append({"pos": int(idx), "time": time.perf_counter() - iter_start})

                    if decode_iterator is not None:
                        decode_iterator.close()

                    decoded_text = self.tokenizer.decode(decoded_ids[:-1], skip_special_tokens=True)
        finally:
            if diag_handle is not None:
                diag_handle.close()
            if memory_demo:
                memory_rows = self._stop_memory_monitor(monitor_state, speed_rows)
                self._write_csv_rows(memory_csv_path, memory_rows)

        if speed_demo:
            self._write_csv_rows(speed_csv_path, speed_rows)
        if demo:
            self._write_csv_rows(demo_csv_path, demo_rows)
            self._maybe_write_divergence_rows(
                demo_csv_path=demo_csv_path,
                decoded_ids=decoded_ids,
                reference_token_ids=reference_token_ids,
                divergence_window=divergence_window,
            )

        return decoded_text
