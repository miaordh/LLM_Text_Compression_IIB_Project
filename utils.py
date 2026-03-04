from typing import List
from decimal import Decimal, getcontext, ROUND_FLOOR, InvalidOperation

def counts_to_cumulative_ascending(counts: list[int]) -> list[int]:
    """Return ascending cumulative array: cum[0]=0, cum[n]=total."""
    cum = [0]
    s = 0
    for c in counts:
        s += c
        cum.append(s)
    return cum



import torch
import numpy as np

def probs_to_counts(probs: np.ndarray, total: int, dec_prec: int = 0) -> List[int]:
    """Convert probabilities to integer counts summing exactly to total."""
    if total <= 0:
        raise ValueError("total must be positive")

    ps = np.atleast_1d(probs).astype(np.float64)
    if ps.ndim != 1:
        ps = ps.reshape(-1)

    ps = np.nan_to_num(ps, nan=0.0, posinf=0.0, neginf=0.0)
    ps = np.clip(ps, 0.0, None)

    s = ps.sum()
    if s <= 0:
        ps = np.full_like(ps, 1.0 / len(ps), dtype=np.float64)
    else:
        ps = ps / s

    counts = np.floor(ps * total).astype(np.int64)

    if total >= len(counts):
        min_req = np.ones_like(counts)
    else:
        min_req = (ps > 0).astype(np.int64)

    counts = np.maximum(counts, min_req)

    diff = int(counts.sum() - total)

    if diff > 0:
        order = np.argsort(counts, kind="stable")[::-1]
        for idx in order:
            if diff == 0:
                break
            spare = int(counts[idx] - min_req[idx])
            if spare <= 0:
                continue
            take = min(spare, diff)
            counts[idx] -= take
            diff -= take
    elif diff < 0:
        errors = (ps * total) - counts
        order = np.argsort(errors, kind="stable")[::-1]
        to_add = -diff
        for idx in order:
            if to_add == 0:
                break
            counts[idx] += 1
            to_add -= 1

    final_sum = int(counts.sum())
    if final_sum != total:
        raise RuntimeError(f"counts sum {final_sum} != total {total}")

    return counts.tolist()

def probs_to_counts_legacy(probs: List[float], total: int, dec_prec: int = 200) -> List[int]:
    """
    Convert probs -> integer counts summing to total using Decimal arithmetic.
    - probs: list of nonnegative floats (may be raw python floats or Decimal-compatible strings)
    - total: integer target sum (can be 2**128 - 1, etc.)
    - dec_prec: decimal precision in *decimal digits*; must be large enough to exactly represent `total`
      and to resolve fractional parts (set to ~bits/3 + 10 digits as heuristic). For 128-bit totals,
      50-80 digits is usually enough; default 200 is conservative.
    Returns: list of Python ints (len == len(probs)) summing to total.
    Deterministic tie-breaking: uses index order for equal fractional parts.
    """
    if total <= 0:
        raise ValueError("total must be positive")

    n = len(probs)
    old_prec = getcontext().prec
    getcontext().prec = dec_prec

    def to_decimal(value) -> Decimal:
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal.from_float(float(value))
        except (TypeError, ValueError, InvalidOperation):
            return Decimal(str(value))

    try:
        dec_probs = [to_decimal(p) for p in probs]
        s = sum(dec_probs)
        if s <= 0:
            raise ValueError("probs sum to zero")
        dec_probs = [p / s for p in dec_probs]

        dec_total = Decimal(total)
        scaled = [p * dec_total for p in dec_probs]
        floors = [int(x.to_integral_value(rounding=ROUND_FLOOR)) for x in scaled]
        fracs = [scaled[i] - Decimal(floors[i]) for i in range(n)]

        pos_idx = [i for i, p in enumerate(dec_probs) if p > 0]
        if len(pos_idx) > total:
            raise ValueError("total too small to give each positive-prob symbol one count")
        min_req = [0] * n
        for i in pos_idx:
            min_req[i] = 1

        for i in range(n):
            if floors[i] < min_req[i]:
                floors[i] = min_req[i]
                fracs[i] = Decimal(0)

        remainder = total - sum(floors)

        if remainder > 0:
            order = sorted(range(n), key=lambda i: (-fracs[i], i))
            idx = 0
            while remainder > 0 and idx < len(order):
                floors[order[idx]] += 1
                remainder -= 1
                idx += 1
        elif remainder < 0:
            deficit = -remainder
            order = sorted(range(n), key=lambda i: (fracs[i], i))
            for i in order:
                if deficit == 0:
                    break
                spare = floors[i] - min_req[i]
                if spare <= 0:
                    continue
                take = min(spare, deficit)
                floors[i] -= take
                deficit -= take
            if deficit != 0:
                raise RuntimeError("Unable to adjust counts to match total; total too small for min requirements")

        final_sum = sum(floors)
        if final_sum != total:
            diff = total - final_sum
            if diff > 0:
                order = sorted(range(n), key=lambda i: (-fracs[i], i))
                idx = 0
                while diff > 0 and idx < len(order):
                    floors[order[idx]] += 1
                    diff -= 1
                    idx += 1
            elif diff < 0:
                deficit = -diff
                order = sorted(range(n), key=lambda i: (fracs[i], i))
                for i in order:
                    if deficit == 0:
                        break
                    spare = floors[i] - min_req[i]
                    if spare <= 0:
                        continue
                    take = min(spare, deficit)
                    floors[i] -= take
                    deficit -= take
                if deficit != 0:
                    raise RuntimeError("Unable to adjust counts after correction; please increase precision")

        final_sum = sum(floors)
        if final_sum != total:
            raise AssertionError(f"counts sum {final_sum} != total {total}")

        return [int(x) for x in floors]
    finally:
        getcontext().prec = old_prec



# def probs_to_counts_largest_remainder(probs: List[float], slots: int) -> List[int]:
#         raw = [float(p) for p in probs]
#         ssum = sum(raw)
#         scaled = [r * slots / ssum for r in raw]
#         floors = [int(x) for x in scaled]
#         remainder = slots - sum(floors)
#         fracs = sorted(((i, scaled[i] - floors[i]) for i in range(len(probs))),
#                        key=lambda x: x[1], reverse=True)
#         i = 0
#         while remainder > 0 and i < len(fracs):
#             floors[fracs[i][0]] += 1
#             remainder -= 1
#             i += 1
#         for p, c in zip(raw, floors):
#             if p > 0.0 and c == 0:
#                 raise ValueError("Precision too low")
#         return floors

def counts_to_cum_desc(counts: List[int]) -> List[int]:
    total = sum(counts)
    cum = [total]
    s = 0
    for c in counts:
        s += c
        cum.append(total - s)
    return cum

def get_context_slice(idx, model, token_ids, context_window=None):
    """Return the portion of token_ids used as context for predicting position idx."""
    max_positions = getattr(model.config, "max_position_embeddings", None)
    start = 0
    if max_positions is not None and idx > max_positions - 1:
        start = idx - (max_positions - 1)
    if context_window is not None:
        if context_window <= 0:
            raise ValueError("context_window must be positive")
        start = max(start, idx - context_window)
    return token_ids[start:idx]

def get_context_slice(idx, model, token_ids):
    """Return the portion of token_ids used as context for predicting position idx."""
    max_positions = getattr(model.config, "max_position_embeddings", None)
    start = 0
    if max_positions is not None and idx > max_positions - 1:
        start = idx - (max_positions - 1)
    return token_ids[start:idx]
