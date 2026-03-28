import csv
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class DiagnosticsWriter:
    """Handles per-token diagnostic CSV writing for encoding/decoding."""

    def __init__(self, prefix: Optional[str], tokenizer):
        self.prefix = prefix
        self.tokenizer = tokenizer

    def open_writer(self, phase: str) -> Tuple[Optional[Any], Optional[Any]]:
        if not self.prefix:
            return None, None
        path = self._diagnostics_csv_path(phase)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=self._diagnostics_fieldnames())
        writer.writeheader()
        return handle, writer

    def write_token_diagnostic(self, writer, phase: str, step_idx: int, token_id: int, raw_probs: np.ndarray, effective_probs: np.ndarray, counts: np.ndarray, coder_L_before: int, coder_R_before: int, coder_D_before: Optional[int] = None):
        if writer is None: return

        token_id = int(token_id)
        counts_np = np.asarray(counts, dtype=np.int64)

        raw_lower, raw_upper, raw_low, raw_high = self._interval_bounds_from_probs(raw_probs, token_id, coder_L_before, coder_R_before)
        eff_lower, eff_upper, eff_low, eff_high, symbol_count, counts_total = self._interval_bounds_from_counts(counts_np, token_id, coder_L_before, coder_R_before)

        prev_token_id = token_id - 1 if token_id > 0 else None
        next_token_id = token_id + 1 if token_id < (len(raw_probs) - 1) else None

        def _neighbor_bounds(neighbor_id: Optional[int]):
            if neighbor_id is None: return None
            n_raw_lower, n_raw_upper, n_raw_low, n_raw_high = self._interval_bounds_from_probs(raw_probs, neighbor_id, coder_L_before, coder_R_before)
            n_eff_lower, n_eff_upper, n_eff_low, n_eff_high, _, _ = self._interval_bounds_from_counts(counts_np, neighbor_id, coder_L_before, coder_R_before)
            return {
                "token_id": neighbor_id,
                "token_text": token_text_for_diag(self.tokenizer, neighbor_id),
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

        writer.writerow({
            "phase": phase,
            "step_idx": int(step_idx),
            "token_id": token_id,
            "token_text": token_text_for_diag(self.tokenizer, token_id),
            "coder_D_before": "" if coder_D_before is None else int(coder_D_before),
            "selected_interval_contains_D": contains_d,
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
        })

    def _diagnostics_csv_path(self, phase: str) -> Path:
        base = Path(str(self.prefix))
        if base.suffix.lower() == ".csv":
            return base.with_name(f"{base.stem}_{phase}.csv")
        return Path(str(base) + f"_{phase}.csv")

    def _diagnostics_fieldnames(self) -> List[str]:
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

    @staticmethod
    def _interval_bounds_from_probs(probs: np.ndarray, token_id: int, coder_L: int, coder_R: int):
        token_id = int(token_id)
        lower_prob = float(np.sum(probs[:token_id]))
        token_prob = float(probs[token_id])
        upper_prob = lower_prob + token_prob

        low = int(coder_L + math.floor(coder_R * lower_prob))
        high = int(coder_L + math.floor(coder_R * upper_prob) - 1)
        return lower_prob, upper_prob, low, high

    @staticmethod
    def _interval_bounds_from_counts(counts: np.ndarray, token_id: int, coder_L: int, coder_R: int):
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

    @staticmethod
    def _rank_from_probs(probs: np.ndarray, token_id: int) -> int:
        token_prob = float(probs[int(token_id)])
        return int(np.count_nonzero(probs > token_prob) + 1)

def token_text_for_diag(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")
    except Exception:
        return ""


def write_csv_rows(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def start_memory_monitor(enabled: bool, sample_interval: float):
    if not enabled:
        return None

    try:
        import psutil
        import torch
    except Exception:
        return None

    stop_event = threading.Event()
    rows: List[Dict[str, float]] = []
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    has_cuda = getattr(torch, "cuda", None) and torch.cuda.is_available()
    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    mps_alloc_fn = None
    mps_driver_alloc_fn = None
    if has_mps and hasattr(torch, "mps"):
        mps_alloc_fn = getattr(torch.mps, "current_allocated_memory", None)
        mps_driver_alloc_fn = getattr(torch.mps, "driver_allocated_memory", None)


    def _monitor():
        while not stop_event.is_set():
            try:
                mem_info = process.memory_info()
                mem_full_info = None
                try:
                    mem_full_info = process.memory_full_info()
                except Exception:
                    mem_full_info = None

                sys_mem = None
                sys_swap = None
                try:
                    sys_mem = psutil.virtual_memory()
                    sys_swap = psutil.swap_memory()
                except Exception:
                    sys_mem = None
                    sys_swap = None

                cuda_allocated_mb = ""
                cuda_reserved_mb = ""
                mps_allocated_mb = ""
                mps_driver_allocated_mb = ""
                vram_mb = ""
                process_uss_mb = ""
                process_swap_mb = ""
                system_used_mb = ""
                system_available_mb = ""
                system_memory_percent = ""
                system_swap_used_mb = ""

                if mem_full_info is not None:
                    if hasattr(mem_full_info, "uss"):
                        process_uss_mb = float(mem_full_info.uss) / (1024.0 * 1024.0)
                    if hasattr(mem_full_info, "swap"):
                        process_swap_mb = float(mem_full_info.swap) / (1024.0 * 1024.0)

                if sys_mem is not None:
                    system_used_mb = float(sys_mem.used) / (1024.0 * 1024.0)
                    system_available_mb = float(sys_mem.available) / (1024.0 * 1024.0)
                    system_memory_percent = float(sys_mem.percent)
                if sys_swap is not None:
                    system_swap_used_mb = float(sys_swap.used) / (1024.0 * 1024.0)

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

                if has_mps and mps_driver_alloc_fn is not None:
                    try:
                        mps_driver_allocated_mb = float(mps_driver_alloc_fn()) / (1024.0 * 1024.0)
                    except Exception:
                        pass

                rows.append(
                    {
                        "time": time.perf_counter() - start_time,
                        "rss_mb": float(mem_info.rss) / (1024.0 * 1024.0),
                        "vms_mb": float(mem_info.vms) / (1024.0 * 1024.0),
                        "process_uss_mb": process_uss_mb,
                        "process_swap_mb": process_swap_mb,
                        "vram_mb": vram_mb,
                        "cuda_allocated_mb": cuda_allocated_mb,
                        "cuda_reserved_mb": cuda_reserved_mb,
                        "mps_allocated_mb": mps_allocated_mb,
                        "mps_driver_allocated_mb": mps_driver_allocated_mb,
                        "system_used_mb": system_used_mb,
                        "system_available_mb": system_available_mb,
                        "system_memory_percent": system_memory_percent,
                        "system_swap_used_mb": system_swap_used_mb,
                    }
                )
            except Exception:
                break
            time.sleep(max(0.01, float(sample_interval)))

    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()
    return {"stop_event": stop_event, "thread": thread, "rows": rows}


def stop_memory_monitor(state: Optional[Dict[str, Any]], speed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if state is None:
        return []

    state["stop_event"].set()
    state["thread"].join()
    rows = state["rows"]
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


def maybe_write_divergence_rows(
    tokenizer,
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
                    "reference_token_text": token_text_for_diag(tokenizer, ref_id),
                    "decoded_token_id": dec_id,
                    "decoded_token_text": token_text_for_diag(tokenizer, dec_id),
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
                    "reference_token_text": "" if ref_id is None else token_text_for_diag(tokenizer, ref_id),
                    "decoded_token_id": "" if dec_id is None else dec_id,
                    "decoded_token_text": "" if dec_id is None else token_text_for_diag(tokenizer, dec_id),
                    "match": int(ref_id is not None and dec_id is not None and ref_id == dec_id),
                    "note": "",
                }
            )

    divergence_path = Path(demo_csv_path)
    divergence_path = divergence_path.with_name(f"{divergence_path.stem}_divergence.csv")
    write_csv_rows(str(divergence_path), rows)

