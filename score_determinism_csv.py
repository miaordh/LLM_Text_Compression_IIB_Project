import argparse
import math
from pathlib import Path
from typing import List, Tuple

import pandas as pd


SCORE_METRICS: List[Tuple[str, float]] = [
    ("max_distance_D_to_reference_interval", 0.35),
    ("mean_distance_D_to_reference_interval", 0.25),
    ("mean_abs_interval_low_delta", 0.10),
    ("mean_abs_interval_high_delta", 0.10),
    ("drift_events", 0.10),
    ("corrections_applied", 0.10),
]


def attach_determinism_score(df: pd.DataFrame) -> pd.DataFrame:
    if "status" not in df.columns:
        raise ValueError("CSV is missing required column: status")

    valid_mask = df["status"].astype(str).eq("ok")
    quality_sum = pd.Series(0.0, index=df.index)

    for metric_name, weight in SCORE_METRICS:
        metric_raw = pd.to_numeric(df.get(metric_name), errors="coerce")
        metric_log = metric_raw.clip(lower=0).map(lambda v: float("nan") if pd.isna(v) else float(math.log1p(v)))

        valid_metric = metric_log[valid_mask]
        min_v = valid_metric.min(skipna=True)
        max_v = valid_metric.max(skipna=True)

        if pd.isna(min_v) or pd.isna(max_v):
            metric_quality = pd.Series(0.0, index=df.index)
        elif max_v > min_v:
            metric_quality = 1.0 - ((metric_log - min_v) / (max_v - min_v))
            metric_quality = metric_quality.clip(lower=0.0, upper=1.0).fillna(0.0)
        else:
            metric_quality = pd.Series(1.0, index=df.index)

        quality_sum += weight * metric_quality

    if "decoded_match" in df.columns:
        decoded_match = pd.to_numeric(df["decoded_match"], errors="coerce").fillna(0.0)
    else:
        # Older CSVs: infer decode match from decoded/original char counts when possible.
        decoded_chars = pd.to_numeric(df.get("decoded_chars"), errors="coerce")
        original_chars = pd.to_numeric(df.get("original_chars"), errors="coerce")
        decoded_match = (decoded_chars == original_chars).astype(float).fillna(0.0)

    if "zero_recenter" in df.columns:
        zero_recenter = pd.to_numeric(df["zero_recenter"], errors="coerce").fillna(0.0)
    else:
        corr = pd.to_numeric(df.get("corrections_applied"), errors="coerce")
        zero_recenter = (corr == 0).astype(float).fillna(0.0)

    strict_gate = ((decoded_match >= 1.0) & (zero_recenter >= 1.0)).astype(float)
    base_score = 100.0 * quality_sum.clip(lower=0.0, upper=1.0)
    final_score = base_score * (0.35 + 0.65 * strict_gate)
    final_score = final_score.where(valid_mask, 0.0)

    out = df.copy()
    if "determinism_mode" in out.columns:
        out["determinism_mode"] = out["determinism_mode"].fillna("None")
    out["determinism_score"] = final_score.round(4)
    out["determinism_tier"] = pd.cut(
        out["determinism_score"],
        bins=[-0.001, 20, 40, 60, 80, 100],
        labels=["very_low", "low", "medium", "high", "very_high"],
    ).astype("string")
    out["determinism_rank"] = out["determinism_score"].rank(method="min", ascending=False).astype("Int64")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Add comparable determinism score columns to a drift-search CSV.")
    parser.add_argument("--input", required=True, help="Path to source CSV")
    parser.add_argument("--output", default=None, help="Path to output CSV (default: <input>_scored.csv)")
    parser.add_argument("--print-top", type=int, default=10, help="Print top-N trials by determinism score")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = input_path.with_name(f"{input_path.stem}_scored{input_path.suffix}")

    df = pd.read_csv(input_path)
    scored = attach_determinism_score(df)
    scored.to_csv(output_path, index=False)

    print(f"Wrote scored CSV: {output_path}")

    cols = [
        c
        for c in [
            "trial_id",
            "status",
            "determinism_mode",
            "quant",
            "slots",
            "logit_round_decimals",
            "prob_round_decimals",
            "drift_events",
            "corrections_applied",
            "determinism_score",
            "determinism_tier",
            "determinism_rank",
        ]
        if c in scored.columns
    ]

    top_n = max(1, int(args.print_top))
    print(scored.sort_values("determinism_score", ascending=False).head(top_n)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
