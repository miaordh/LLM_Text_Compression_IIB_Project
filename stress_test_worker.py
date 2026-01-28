
import sys
import time
import psutil
import os
import pandas as pd
import torch
import gc
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import local class
try:
    from llm_codec import LLM_Encoder
except ImportError:
    sys.exit(1)

def run_stress_test(strategy, device, filename, limit, output_id):
    # --- 1. SETUP MONITORING ---
    memory_log = []
    stop_monitor = False
    process = psutil.Process(os.getpid())
    start_time = time.time()

    def monitor():
        while not stop_monitor:
            # RSS = Resident Set Size (Physical RAM) in MB
            mem_mb = process.memory_info().rss / (1024 * 1024)
            elapsed = time.time() - start_time
            memory_log.append({"time": elapsed, "memory_mb": mem_mb})
            time.sleep(0.1)

    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    # --- 2. LOAD & RUN ---
    try:
        model_id = "deepseek-ai/deepseek-coder-1.3b-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()

        # Crop text to limit (approx 4 chars per token)
        safe_text = text[:limit * 4] 

        # Force device placement
        if device == "cpu":
            model.to("cpu")
        elif device == "mps":
            model.to("mps")

        encoder = LLM_Encoder(
            tokenizer=tokenizer,
            model=model,
            strategy=strategy,
            context_window=2048,
            margin=128,
            device=device
        )

        # Use the class's built-in speed tracking
        speed_csv = f"temp_speed_{output_id}.csv"
        encoder.encode(safe_text, demo=False, speed_demo=True, speed_csv_path=speed_csv)

    except Exception as e:
        print(f"Worker Error: {e}")
    finally:
        stop_monitor = True
        monitor_thread.join()

    # --- 3. SAVE MEMORY LOG ---
    pd.DataFrame(memory_log).to_csv(f"temp_memory_{output_id}.csv", index=False)
    sys.exit(0)

if __name__ == "__main__":
    # Arguments: strategy, device, filename, limit(int), output_id
    if len(sys.argv) >= 6:
        run_stress_test(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
