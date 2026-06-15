import csv
import re
import sys
from pathlib import Path

def process(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    added_cols = ["slots", "logit_round_decimals", "prob_round_decimals"]
    for col in added_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    for row in rows:
        artifact_root = row.get("artifact_root", "")
        # Try to parse from artifact root
        # example: .roundtrip_artifacts/crossdev_hf_none_slots1048576_l0_p0_2026-05-16_21-53-59
        
        row["slots"] = ""
        row["logit_round_decimals"] = ""
        row["prob_round_decimals"] = ""

        if "unquant" in artifact_root:
            row["slots"] = "unquant"
            row["logit_round_decimals"] = "unquant"
            row["prob_round_decimals"] = "unquant"
        else:
            m = re.search(r'_slots(\d+)_l(\d+)_p(\d+)_', artifact_root)
            if m:
                row["slots"] = m.group(1)
                row["logit_round_decimals"] = m.group(2)
                row["prob_round_decimals"] = m.group(3)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

process(sys.argv[1])
