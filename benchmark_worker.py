import sys
import time
import pandas as pd
import torch
import gc
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

try:
    from llm_codec import LLM_Encoder
except ImportError:
    print("Error: Could not import LLM_Encoder from llm_codec.py")
    sys.exit(1)

def run_single_benchmark(model_id, display_name, text_label, text_path, csv_path):
    print(f"--- WORKER START: {display_name} on {text_label} ---")

    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"File not found: {text_path}")
        sys.exit(1)

    original_bytes = len(text.encode('utf-8'))

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if hasattr(tokenizer, "model_max_length"):
            tokenizer.model_max_length = 1000000
        model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as e:
        print(f"Model load failed: {e}")
        sys.exit(1)

    # Use "block" for Novel to prevent memory leaks/crash
    strategy = "block" if text_label == "Novel (En)" else "rolling"

    encoder = LLM_Encoder(
        tokenizer=tokenizer,
        model=model,
        precision=32,
        context_window=2048,
        margin=128,
        device="cpu", # Safe CPU
        strategy=strategy
    )

    tik = time.time()
    encoded = encoder.encode(text, demo=False, speed_demo=False)
    enc_time = time.time() - tik
    enc_ratio = (len(encoded) * 8) / original_bytes

    print(f"DONE. Time: {enc_time:.2f}s | Ratio: {enc_ratio:.4f}")

    new_row = {
        "Text": text_label, "Model": display_name, "Time": enc_time, 
        "Ratio": enc_ratio, "Tokens": len(encoded), "Device": "CPU", "Type": "LLM"
    }

    header = not os.path.exists(csv_path)
    pd.DataFrame([new_row]).to_csv(csv_path, mode='a', header=header, index=False)
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(1)
    run_single_benchmark(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
