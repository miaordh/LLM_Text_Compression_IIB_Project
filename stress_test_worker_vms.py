
import sys
import time
import psutil
import os
import pandas as pd
import torch
import gc
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from llm_codec import LLM_Encoder
except ImportError:
    sys.exit(1)

def run_stress_test(strategy, device, filename, limit, window_size, output_id):
    memory_log = []
    stop_monitor = False
    process = psutil.Process(os.getpid())
    start_time = time.time()

    def monitor():
        while not stop_monitor:
            try:
                mem_info = process.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)
                vms_mb = mem_info.vms / (1024 * 1024)
                elapsed = time.time() - start_time
                memory_log.append({"time": elapsed, "rss_mb": rss_mb, "vms_mb": vms_mb})
                time.sleep(0.05)
            except:
                break

    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    try:
        model_id = "deepseek-ai/deepseek-coder-1.3b-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        with open(filename, 'r', encoding='utf-8') as f: full_text = f.read()
        all_ids = tokenizer.encode(full_text)
        if len(all_ids) > limit: all_ids = all_ids[:limit]
        safe_text = tokenizer.decode(all_ids)

        if device == "cpu": model.to("cpu")
        elif device == "mps": model.to("mps")

        encoder = LLM_Encoder(
            tokenizer=tokenizer, model=model, strategy=strategy,
            context_window=window_size, margin=32, device=device
        )

        speed_csv = f"temp_speed_{output_id}.csv"
        encoder.encode(safe_text, demo=False, speed_demo=True, speed_csv_path=speed_csv)

    except Exception as e:
        print(f"Worker Error: {e}")
    finally:
        stop_monitor = True
        monitor_thread.join()

    pd.DataFrame(memory_log).to_csv(f"temp_memory_{output_id}.csv", index=False)
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) >= 7:
        run_stress_test(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
