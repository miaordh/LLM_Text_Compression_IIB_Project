import gc
import csv
import importlib
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from arithmetic_coding import Coder
from bitReadWrite import BitReader, BitWriter
from decoder import Decoder
from encoder import Encoder
from utils import counts_to_cum_desc, probs_to_counts, probs_to_counts_legacy

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


class DeterministicLLMCodec:
    def __init__(
        self,
        tokenizer,
        model,
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

        if "<EOF>" not in self.tokenizer.all_special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<EOF>"]})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.eof_token_id = self.tokenizer.convert_tokens_to_ids("<EOF>")

        self.dec_prec = max(50, int(math.ceil(self.config.precision * math.log10(2))) + 10)

        self._batch_invariant_enabled = False
        self._log_softmax_fn = torch.log_softmax
        self._batch_invariant_ctx = nullcontext
        self._determinism_mode = _normalize_determinism_mode(self.config.determinism_mode)

        try:
            self.model.config._attn_implementation = "eager"
        except Exception:
            pass

        self._configure_determinism_runtime()

    def _diagnostics_enabled(self) -> bool:
        return bool(self.config.diagnostics_csv_prefix)

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

    def encode(
        self,
        text: str,
        safe_mode: bool = False,
        return_token_count: bool = False,
        show_progress: bool = True,
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

        try:
            with self._invariant_context():
                if self.use_kv_cache:
                    cache_state = self._hard_reset_cache_and_warmup([], 0, 0)
                    cache_state["next_logits"] = self._get_logits(0, token_ids, cache_state)

                for idx, token_id in iterator:
                    if self.config.strategy == "block" and idx > 0 and (idx % self.block_stride == 0):
                        cache_state = self._hard_reset_cache_and_warmup(
                            token_ids,
                            current_index=idx,
                            warmup_length=self.config.margin,
                        )

                    logits = self._get_logits(idx, token_ids, cache_state)
                    raw_probs, probs = self._raw_and_effective_probs(logits)
                    counts = self._counts_from_probs(probs)

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
        finally:
            if diag_handle is not None:
                diag_handle.close()

        enc.finish()
        writer.flush()
        encoded_bytes = writer.getvalue()

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
    ) -> str:
        dec = Decoder(Coder(b=self.config.precision), BitReader(encoded_bytes))
        decoded_ids = []
        cache_state = self._init_cache_state()
        diag_handle, diag_writer = self._open_diagnostics_writer("decode")

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

                        decoded_ids.append(token_id)

                        cache_state = self._advance_state(token_id, cache_state)
                        if (
                            self.config.strategy == "rolling"
                            and cache_state["cached_token_count"]
                            > (self.config.context_window + self.config.margin)
                        ):
                            cache_state = self._truncate_cache_rolling(cache_state)

                    return self.tokenizer.decode(decoded_ids, skip_special_tokens=True)

                token_id = None
                decode_iterator = None
                if show_progress:
                    decode_iterator = tqdm(desc="Deterministic Decode", unit="tok")
                while token_id != self.eof_token_id:
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

                if decode_iterator is not None:
                    decode_iterator.close()
        finally:
            if diag_handle is not None:
                diag_handle.close()

        return self.tokenizer.decode(decoded_ids[:-1], skip_special_tokens=True)
