import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
FINEZIP_AC_DIR = PROJECT_ROOT / "FineZip" / "AC"
if str(FINEZIP_AC_DIR) not in sys.path:
    sys.path.insert(0, str(FINEZIP_AC_DIR))

from eval_ac import AC_Decode, AC_Encode, read_bitstream, text_to_tokens, verify_text
from arithmeticcoding import BitInputStream


def _compressed_stats(output_dir: Path, batch_size: int):
    compressed_size_bytes = 0
    compressed_bits_count = 0
    for index in range(batch_size):
        ac_path = output_dir / f"{index}_AC.txt"
        compressed_size_bytes += ac_path.stat().st_size
        with ac_path.open("rb") as handle:
            bitin = BitInputStream(handle)
            compressed_bits_count += int(read_bitstream(bitin).size)
    return compressed_size_bytes, compressed_bits_count


def _roundtrip_ok(input_file: Path, decoded_text: str) -> bool:
    source_text = input_file.read_text(encoding="utf-8")
    if decoded_text[:17] == "<|begin_of_text|>":
        decoded_text = decoded_text[17:]
    return source_text == decoded_text


def _write_metrics_csv(output_dir: Path, row: dict):
    metrics_path = output_dir / "metrics.csv"
    fieldnames = [
        "model",
        "tokenizer",
        "input_file",
        "batch_size",
        "context_size",
        "original_size_bytes",
        "compressed_size_bytes",
        "compression_ratio_bits_per_byte",
        "compression_time_seconds",
        "decompression_time_seconds",
        "roundtrip_ok",
        "compressed_bits_count",
        "rho_bits_per_char",
        "num_chars",
        "num_tokens",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})
    print(f"Run metrics written to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Root-level FineZip AC runner with per-run CSV metrics."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--context_size", type=int, required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--AC_output_dir", required=True)
    parser.add_argument("--encode_decode", default="1")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.AC_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model.eval()
    device = torch.cuda.current_device()
    model.to(device)

    text_input = input_file.read_text(encoding="utf-8")
    tokens_full = text_to_tokens(tokenizer, text_input)
    compressed_file_name = str(input_file)[:-4]

    encoder = AC_Encode(
        model,
        tokenizer,
        compressed_file_name,
        str(output_dir),
        device,
        args.batch_size,
        args.context_size,
    )

    encode_start = time.time()
    total_length, pad_len, vocab_size = encoder.encode_from_tokens(tokens_full)
    compression_time = time.time() - encode_start

    encoder.compute_compression_ratio(tokens_full, compression_time, str(input_file))
    compressed_size_bytes, compressed_bits_count = _compressed_stats(output_dir, args.batch_size)

    original_size_bytes = input_file.stat().st_size
    text_encoded = tokenizer.decode(tokens_full.squeeze().tolist())
    num_chars = len(text_encoded)
    num_tokens = int(len(tokens_full))

    metrics = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "input_file": str(input_file),
        "batch_size": int(args.batch_size),
        "context_size": int(args.context_size),
        "original_size_bytes": int(original_size_bytes),
        "compressed_size_bytes": int(compressed_size_bytes),
        "compression_ratio_bits_per_byte": (
            float(compressed_size_bytes * 8 / original_size_bytes)
            if original_size_bytes > 0
            else None
        ),
        "compression_time_seconds": float(compression_time),
        "decompression_time_seconds": None,
        "roundtrip_ok": None,
        "compressed_bits_count": int(compressed_bits_count),
        "rho_bits_per_char": (
            float(compressed_bits_count / num_chars) if num_chars > 0 else None
        ),
        "num_chars": int(num_chars),
        "num_tokens": num_tokens,
    }

    if str(args.encode_decode) == "1":
        decoder = AC_Decode(
            model,
            tokenizer,
            device,
            str(output_dir),
            args.batch_size,
            args.context_size,
        )
        decode_start = time.time()
        decoded_text_ac = decoder.decode_AC(total_length, pad_len, vocab_size)
        metrics["decompression_time_seconds"] = float(time.time() - decode_start)
        verify_text(compressed_file_name, str(input_file), decoded_text_ac)
        metrics["roundtrip_ok"] = _roundtrip_ok(input_file, decoded_text_ac)

    _write_metrics_csv(output_dir, metrics)


if __name__ == "__main__":
    main()
