import csv
import gc
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from arithmetic_coding import Coder
from bitReadWrite import BitReader, BitWriter
from decoder import Decoder
from encoder import Encoder
from llm_codec_deterministic import DeterministicCodecConfig, DeterministicLLMCodec


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


@dataclass
class DriftTestCodecConfig:
    precision: int = 32
    slots: int = (1 << 24)

    context_window: int = 2048
    margin: int = 128
    strategy: str = "rolling"  # rolling | block | no_kv_cache
    use_legacy_counts: bool = False

    quant: bool = False
    logit_round_decimals: int = 2
    prob_round_decimals: int = 5

    diagnostics_csv_prefix: Optional[str] = None
    determinism_mode: Optional[str] = None
    inference_backend: str = "auto"
    model_id: Optional[str] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None
    torch_dtype: str = "auto"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_logprobs: Optional[int] = None
    vllm_max_model_len: Optional[int] = None

    # Drift-test specific knobs
    drift_correction_enabled: bool = True
    drift_measurements_csv_prefix: Optional[str] = None
    encoder_trace_csv_prefix: Optional[str] = None
    emit_full_reference_trace: bool = True


class DriftAwareLLMCodec:
    """Drift-testing wrapper around DeterministicLLMCodec.

    Features:
    - encode_with_trace: emits bitstream + per-token reference intervals from encoder.
    - decode_with_reference: detects drift versus reference, logs metrics, optionally recenters
      decoder state to reference interval and continues with corrected history.
    """

    def __init__(
        self,
        tokenizer,
        model=None,
        device: str = "auto",
        config: Optional[DriftTestCodecConfig] = None,
    ):
        self.config = config or DriftTestCodecConfig()
        self.base = DeterministicLLMCodec(
            tokenizer=tokenizer,
            model=model,
            device=device,
            config=DeterministicCodecConfig(
                precision=self.config.precision,
                slots=self.config.slots,
                context_window=self.config.context_window,
                margin=self.config.margin,
                strategy=self.config.strategy,
                use_legacy_counts=self.config.use_legacy_counts,
                quant=self.config.quant,
                logit_round_decimals=self.config.logit_round_decimals,
                prob_round_decimals=self.config.prob_round_decimals,
                diagnostics_csv_prefix=self.config.diagnostics_csv_prefix,
                determinism_mode=self.config.determinism_mode,
                inference_backend=getattr(self.config, "inference_backend", "auto"),
                model_id=getattr(self.config, "model_id", None),
                trust_remote_code=bool(getattr(self.config, "trust_remote_code", False)),
                revision=getattr(self.config, "revision", None),
                torch_dtype=str(getattr(self.config, "torch_dtype", "auto")),
                vllm_tensor_parallel_size=int(getattr(self.config, "vllm_tensor_parallel_size", 1)),
                vllm_gpu_memory_utilization=float(getattr(self.config, "vllm_gpu_memory_utilization", 0.9)),
                vllm_max_logprobs=getattr(self.config, "vllm_max_logprobs", None),
                vllm_max_model_len=getattr(self.config, "vllm_max_model_len", None),
            ),
        )
        self.device = self.base.device
        self.tokenizer = self.base.tokenizer
        self.model = self.base.model
        self.eof_token_id = self.base.eof_token_id

    @staticmethod
    def _csv_path(prefix: Optional[str], phase: str) -> Optional[Path]:
        if not prefix:
            return None
        base = Path(str(prefix))
        if base.suffix.lower() == ".csv":
            return base.with_name(f"{base.stem}_{phase}.csv")
        return Path(str(base) + f"_{phase}.csv")

    @staticmethod
    def _write_csv(path: Optional[Path], fieldnames: List[str], rows: List[Dict[str, Any]]):
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _distance_to_interval(value: int, low: int, high: int) -> int:
        if low <= value <= high:
            return 0
        if value < low:
            return low - value
        return value - high

    @staticmethod
    def _select_symbol_and_interval(cum_desc: List[int], coder: Coder) -> Tuple[int, int, int]:
        total = int(cum_desc[0])
        L = int(coder.L)
        R = int(coder.R)
        D = int(coder.D)

        selected = len(cum_desc) - 2
        selected_low = L
        selected_high = L + R - 1

        for s in range(len(cum_desc) - 1):
            l = int(cum_desc[s + 1])
            h = int(cum_desc[s])
            lower = total - h
            upper = total - l
            low = L + (R * lower) // total
            high = L + (R * upper) // total - 1
            if low <= D <= high:
                selected = s
                selected_low = low
                selected_high = high
                break

        return selected, int(selected_low), int(selected_high)

    def encode_with_trace(
        self,
        text: str,
        safe_mode: bool = False,
        return_token_count: bool = True,
        show_progress: bool = True,
    ):
        token_ids = self.tokenizer.encode(text)
        if not safe_mode:
            token_ids = token_ids + [self.eof_token_id]

        writer = BitWriter()
        enc = Encoder(Coder(b=self.config.precision), writer)

        cache_state = self.base._init_cache_state()
        reference_rows: List[Dict[str, Any]] = []

        iterator = enumerate(token_ids)
        if show_progress:
            iterator = tqdm(iterator, total=len(token_ids), desc="DriftTest Encode")

        with self.base._invariant_context():
            if self.base.use_kv_cache:
                cache_state = self.base._hard_reset_cache_and_warmup([], 0, 0)
                cache_state["next_logits"] = self.base._get_logits(0, token_ids, cache_state)

            for idx, token_id in iterator:
                if self.config.strategy == "block" and idx > 0 and (idx % self.base.block_stride == 0):
                    cache_state = self.base._hard_reset_cache_and_warmup(
                        token_ids,
                        current_index=idx,
                        warmup_length=self.config.margin,
                    )

                logits = self.base._get_logits(idx, token_ids, cache_state)
                raw_probs, effective_probs = self.base._raw_and_effective_probs(logits)
                counts = self.base._counts_from_probs(effective_probs)
                cum_desc = counts_to_cum_desc(counts)

                coder_L_before = int(enc.coder.L)
                coder_R_before = int(enc.coder.R)
                _, _, enc_low, enc_high, symbol_count, counts_total = self.base._interval_bounds_from_counts(
                    counts,
                    token_id,
                    coder_L_before,
                    coder_R_before,
                )

                reference_rows.append(
                    {
                        "step_idx": idx,
                        "token_id": int(token_id),
                        "token_text": self.base._token_text_for_diag(self.tokenizer, int(token_id)),
                        "coder_L_before": coder_L_before,
                        "coder_R_before": coder_R_before,
                        "enc_interval_low": int(enc_low),
                        "enc_interval_high": int(enc_high),
                        "symbol_count": int(symbol_count),
                        "counts_total": int(counts_total),
                        "top1_effective_token_id": int(np.argmax(effective_probs)),
                        "top1_effective_prob": float(np.max(effective_probs)),
                        "selected_effective_prob": float(effective_probs[int(token_id)]),
                        "selected_effective_rank": int(self.base._rank_from_probs(effective_probs, int(token_id))),
                        "determinism_mode": self.config.determinism_mode,
                    }
                )

                enc.encode_symbol(int(token_id), cum_desc)

                if idx < len(token_ids) - 1:
                    cache_state = self.base._advance_state(int(token_id), cache_state)
                    if (
                        self.config.strategy == "rolling"
                        and cache_state["cached_token_count"] > (self.config.context_window + self.config.margin)
                    ):
                        cache_state = self.base._truncate_cache_rolling(cache_state)

        enc.finish()
        writer.flush()
        encoded_bytes = writer.getvalue()

        trace_path = self._csv_path(self.config.encoder_trace_csv_prefix, "encode_trace")
        if self.config.emit_full_reference_trace:
            self._write_csv(
                trace_path,
                [
                    "step_idx",
                    "token_id",
                    "token_text",
                    "coder_L_before",
                    "coder_R_before",
                    "enc_interval_low",
                    "enc_interval_high",
                    "symbol_count",
                    "counts_total",
                    "top1_effective_token_id",
                    "top1_effective_prob",
                    "selected_effective_prob",
                    "selected_effective_rank",
                    "determinism_mode",
                ],
                reference_rows,
            )

        if safe_mode or return_token_count:
            return encoded_bytes, len(token_ids), reference_rows
        return encoded_bytes, None, reference_rows

    def decode_with_reference(
        self,
        encoded_bytes: bytes,
        reference_rows: List[Dict[str, Any]],
        max_decode_tokens: Optional[int] = None,
        safe_mode: bool = False,
        expected_num_tokens: Optional[int] = None,
        show_progress: bool = True,
    ):
        dec = Decoder(Coder(b=self.config.precision), BitReader(encoded_bytes))
        decoded_ids: List[int] = []
        drift_rows: List[Dict[str, Any]] = []
        cache_state = self.base._init_cache_state()

        if safe_mode and expected_num_tokens is None:
            raise ValueError("safe_mode=True requires expected_num_tokens")

        def _append_drift_row(
            idx: int,
            coder_D_before: int,
            predicted_token: int,
            applied_token: int,
            pred_low: int,
            pred_high: int,
            ref_token: Optional[int],
            ref_low: Optional[int],
            ref_high: Optional[int],
            effective_probs: np.ndarray,
        ):
            drifted = (ref_token is not None) and (predicted_token != ref_token)
            corrected = drifted and (applied_token == ref_token)
            ref_rank = ""
            if ref_token is not None and 0 <= int(ref_token) < len(effective_probs):
                ref_rank = int(self.base._rank_from_probs(effective_probs, int(ref_token)))
            drift_rows.append(
                {
                    "step_idx": idx,
                    "coder_D_before": int(coder_D_before),
                    "predicted_token_id": int(predicted_token),
                    "applied_token_id": int(applied_token),
                    "reference_token_id": "" if ref_token is None else int(ref_token),
                    "pred_interval_low": int(pred_low),
                    "pred_interval_high": int(pred_high),
                    "reference_interval_low": "" if ref_low is None else int(ref_low),
                    "reference_interval_high": "" if ref_high is None else int(ref_high),
                    "pred_interval_contains_D": int(pred_low <= coder_D_before <= pred_high),
                    "reference_interval_contains_D": ""
                    if ref_low is None or ref_high is None
                    else int(ref_low <= coder_D_before <= ref_high),
                    "distance_D_to_reference_interval": ""
                    if ref_low is None or ref_high is None
                    else int(self._distance_to_interval(int(coder_D_before), int(ref_low), int(ref_high))),
                    "interval_low_delta_pred_minus_ref": ""
                    if ref_low is None
                    else int(pred_low - int(ref_low)),
                    "interval_high_delta_pred_minus_ref": ""
                    if ref_high is None
                    else int(pred_high - int(ref_high)),
                    "predicted_token_prob": float(effective_probs[int(predicted_token)]),
                    "applied_token_prob": float(effective_probs[int(applied_token)]),
                    "reference_token_prob": ""
                    if ref_token is None
                    else float(effective_probs[int(ref_token)]),
                    "predicted_token_rank": int(self.base._rank_from_probs(effective_probs, int(predicted_token))),
                    "applied_token_rank": int(self.base._rank_from_probs(effective_probs, int(applied_token))),
                    "reference_token_rank": ref_rank,
                    "drifted": int(drifted),
                    "corrected": int(corrected),
                }
            )

        with self.base._invariant_context():
            if self.base.use_kv_cache:
                cache_state = self.base._hard_reset_cache_and_warmup([], 0, 0)
                cache_state["next_logits"] = self.base._get_logits(0, [], cache_state)

            if safe_mode:
                target_tokens = int(expected_num_tokens)
                iterator = range(target_tokens)
                if show_progress:
                    iterator = tqdm(iterator, total=target_tokens, desc="DriftTest Decode")

                for _ in iterator:
                    idx = len(decoded_ids)
                    if max_decode_tokens is not None and idx >= max_decode_tokens:
                        raise RuntimeError("max_decode_tokens reached before expected_num_tokens")

                    if self.config.strategy == "block" and idx > 0 and (idx % self.base.block_stride == 0):
                        cache_state = self.base._hard_reset_cache_and_warmup(
                            decoded_ids,
                            current_index=idx,
                            warmup_length=self.config.margin,
                        )

                    logits = self.base._get_logits(idx, decoded_ids, cache_state)
                    _, effective_probs = self.base._raw_and_effective_probs(logits)
                    counts = self.base._counts_from_probs(effective_probs)
                    cum_desc = counts_to_cum_desc(counts)

                    coder_D_before = int(dec.coder.D)
                    predicted_token, pred_low, pred_high = self._select_symbol_and_interval(cum_desc, dec.coder)

                    ref_token = None
                    ref_low = None
                    ref_high = None
                    if idx < len(reference_rows):
                        ref_token = int(reference_rows[idx]["token_id"])
                        ref_low = int(reference_rows[idx]["enc_interval_low"])
                        ref_high = int(reference_rows[idx]["enc_interval_high"])

                    use_reference = bool(
                        self.config.drift_correction_enabled
                        and ref_token is not None
                        and predicted_token != ref_token
                    )

                    if use_reference:
                        dec.coder.set_interval_and_renorm_decode(int(ref_low), int(ref_high))
                        applied_token = int(ref_token)
                    else:
                        dec.coder.set_interval_and_renorm_decode(int(pred_low), int(pred_high))
                        applied_token = int(predicted_token)

                    _append_drift_row(
                        idx=idx,
                        coder_D_before=coder_D_before,
                        predicted_token=int(predicted_token),
                        applied_token=int(applied_token),
                        pred_low=int(pred_low),
                        pred_high=int(pred_high),
                        ref_token=ref_token,
                        ref_low=ref_low,
                        ref_high=ref_high,
                        effective_probs=effective_probs,
                    )

                    decoded_ids.append(int(applied_token))

                    cache_state = self.base._advance_state(int(applied_token), cache_state)
                    if (
                        self.config.strategy == "rolling"
                        and cache_state["cached_token_count"] > (self.config.context_window + self.config.margin)
                    ):
                        cache_state = self.base._truncate_cache_rolling(cache_state)

                decoded_text = self.tokenizer.decode(decoded_ids, skip_special_tokens=True)
            else:
                token_id = None
                decode_iterator = None
                if show_progress:
                    decode_iterator = tqdm(desc="DriftTest Decode", unit="tok")

                while token_id != self.eof_token_id:
                    idx = len(decoded_ids)
                    if max_decode_tokens is not None and idx >= max_decode_tokens:
                        raise RuntimeError("max_decode_tokens reached before EOF")

                    if self.config.strategy == "block" and idx > 0 and (idx % self.base.block_stride == 0):
                        cache_state = self.base._hard_reset_cache_and_warmup(
                            decoded_ids,
                            current_index=idx,
                            warmup_length=self.config.margin,
                        )

                    logits = self.base._get_logits(idx, decoded_ids, cache_state)
                    _, effective_probs = self.base._raw_and_effective_probs(logits)
                    counts = self.base._counts_from_probs(effective_probs)
                    cum_desc = counts_to_cum_desc(counts)

                    coder_D_before = int(dec.coder.D)
                    predicted_token, pred_low, pred_high = self._select_symbol_and_interval(cum_desc, dec.coder)

                    ref_token = None
                    ref_low = None
                    ref_high = None
                    if idx < len(reference_rows):
                        ref_token = int(reference_rows[idx]["token_id"])
                        ref_low = int(reference_rows[idx]["enc_interval_low"])
                        ref_high = int(reference_rows[idx]["enc_interval_high"])

                    use_reference = bool(
                        self.config.drift_correction_enabled
                        and ref_token is not None
                        and predicted_token != ref_token
                    )

                    if use_reference:
                        dec.coder.set_interval_and_renorm_decode(int(ref_low), int(ref_high))
                        applied_token = int(ref_token)
                    else:
                        dec.coder.set_interval_and_renorm_decode(int(pred_low), int(pred_high))
                        applied_token = int(predicted_token)

                    _append_drift_row(
                        idx=idx,
                        coder_D_before=coder_D_before,
                        predicted_token=int(predicted_token),
                        applied_token=int(applied_token),
                        pred_low=int(pred_low),
                        pred_high=int(pred_high),
                        ref_token=ref_token,
                        ref_low=ref_low,
                        ref_high=ref_high,
                        effective_probs=effective_probs,
                    )

                    decoded_ids.append(int(applied_token))
                    token_id = int(applied_token)

                    if token_id != self.eof_token_id:
                        cache_state = self.base._advance_state(token_id, cache_state)
                        if (
                            self.config.strategy == "rolling"
                            and cache_state["cached_token_count"] > (self.config.context_window + self.config.margin)
                        ):
                            cache_state = self.base._truncate_cache_rolling(cache_state)

                    if decode_iterator is not None:
                        decode_iterator.update(1)

                if decode_iterator is not None:
                    decode_iterator.close()

                decoded_text = self.tokenizer.decode(decoded_ids[:-1], skip_special_tokens=True)

        drift_path = self._csv_path(self.config.drift_measurements_csv_prefix, "decode_drift")
        self._write_csv(
            drift_path,
            [
                "step_idx",
                "coder_D_before",
                "predicted_token_id",
                "applied_token_id",
                "reference_token_id",
                "pred_interval_low",
                "pred_interval_high",
                "reference_interval_low",
                "reference_interval_high",
                "pred_interval_contains_D",
                "reference_interval_contains_D",
                "distance_D_to_reference_interval",
                "interval_low_delta_pred_minus_ref",
                "interval_high_delta_pred_minus_ref",
                "predicted_token_prob",
                "applied_token_prob",
                "reference_token_prob",
                "predicted_token_rank",
                "applied_token_rank",
                "reference_token_rank",
                "drifted",
                "corrected",
            ],
            drift_rows,
        )

        return decoded_text, drift_rows


def summarize_drift_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "drift_events": 0,
            "corrections_applied": 0,
            "mean_abs_interval_low_delta": 0.0,
            "mean_abs_interval_high_delta": 0.0,
            "mean_distance_D_to_reference_interval": 0.0,
            "max_distance_D_to_reference_interval": 0,
        }

    drift_events = sum(int(r.get("drifted", 0)) for r in rows)
    corrections = sum(int(r.get("corrected", 0)) for r in rows)

    low_deltas = [
        abs(int(r["interval_low_delta_pred_minus_ref"]))
        for r in rows
        if str(r.get("interval_low_delta_pred_minus_ref", "")) != ""
    ]
    high_deltas = [
        abs(int(r["interval_high_delta_pred_minus_ref"]))
        for r in rows
        if str(r.get("interval_high_delta_pred_minus_ref", "")) != ""
    ]
    distances = [
        int(r["distance_D_to_reference_interval"])
        for r in rows
        if str(r.get("distance_D_to_reference_interval", "")) != ""
    ]

    return {
        "drift_events": int(drift_events),
        "corrections_applied": int(corrections),
        "mean_abs_interval_low_delta": float(np.mean(low_deltas)) if low_deltas else 0.0,
        "mean_abs_interval_high_delta": float(np.mean(high_deltas)) if high_deltas else 0.0,
        "mean_distance_D_to_reference_interval": float(np.mean(distances)) if distances else 0.0,
        "max_distance_D_to_reference_interval": int(np.max(distances)) if distances else 0,
    }


def cleanup_codec(codec: Optional[DriftAwareLLMCodec]):
    if codec is None:
        return
    try:
        del codec
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
