import argparse
import csv
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Optional


PROJECT_DIR_NAME = "LLM_Text_Compression_IIB_Project"


def _job_id(path_text: str) -> str:
    path = Path(path_text)
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}_{digest}"


def _count_positionwise_matching_characters(left: str, right: str) -> int:
    count = 0
    for a, b in zip(left, right):
        if a == b:
            count += 1
        else:
            break
    return count


def _candidate_project_roots(local_project_root: Optional[Path]) -> Iterable[Path]:
    if local_project_root is not None:
        yield local_project_root.resolve()
    yield Path(__file__).resolve().parent


def _map_project_path(path_text: str, local_project_root: Optional[Path]) -> Path:
    path = Path(path_text)
    if path.exists():
        return path

    parts = path.parts
    if PROJECT_DIR_NAME in parts:
        idx = parts.index(PROJECT_DIR_NAME)
        suffix = Path(*parts[idx + 1 :])
        for root in _candidate_project_roots(local_project_root):
            candidate = root / suffix
            if candidate.exists():
                return candidate
    return path


def _decoded_path_for_row(row: Dict[str, str], local_project_root: Optional[Path]) -> Path:
    artifact_root = row.get("trial_artifact_root") or row.get("artifact_root") or ""
    file_path = row.get("file") or ""
    if not artifact_root or not file_path:
        return Path("")

    remote_decoded = Path(artifact_root) / _job_id(file_path) / "decoded.txt"
    if remote_decoded.exists():
        return remote_decoded
    return _map_project_path(str(remote_decoded), local_project_root)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def add_matching_characters(
    input_csv: Path,
    output_csv: Path,
    local_project_root: Optional[Path],
):
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for column in ("matching_characters", "matching_character_error"):
        if column not in fieldnames:
            fieldnames.append(column)

    for row in rows:
        row["matching_characters"] = ""
        row["matching_character_error"] = ""
        status = (row.get("status") or "").strip().lower()
        if status not in {"ok", "mismatch"}:
            continue

        try:
            original_path = _map_project_path(row.get("file", ""), local_project_root)
            decoded_path = _decoded_path_for_row(row, local_project_root)
            if not original_path.exists():
                raise FileNotFoundError(f"original not found: {original_path}")
            if not decoded_path.exists():
                raise FileNotFoundError(f"decoded not found: {decoded_path}")

            original_text = _read_text(original_path)
            decoded_text = _read_text(decoded_path)
            row["matching_characters"] = str(
                _count_positionwise_matching_characters(original_text, decoded_text)
            )
        except Exception as exc:
            row["matching_character_error"] = str(exc)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Add position-wise matching character counts to roundtrip result CSVs."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input>_with_matching.csv unless --in-place is set.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input CSV.",
    )
    parser.add_argument(
        "--local-project-root",
        type=Path,
        default=None,
        help="Local mirror of LLM_Text_Compression_IIB_Project for mapping remote paths.",
    )
    args = parser.parse_args()

    input_csv = args.input_csv.resolve()
    if args.in_place:
        output_csv = input_csv
    elif args.output is not None:
        output_csv = args.output.resolve()
    else:
        output_csv = input_csv.with_name(f"{input_csv.stem}_with_matching{input_csv.suffix}")

    add_matching_characters(input_csv, output_csv, args.local_project_root)
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()
