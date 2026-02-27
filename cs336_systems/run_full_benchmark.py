import subprocess
import re
import pandas as pd
from datetime import datetime

MODEL_SIZES = ["small", "medium", "large", "xl", "2.7B"]
WARMUP_EXPERIMENTS = [0, 1, 2, 5]
LOG_FILE = "benchmark.log"

def run_benchmark(model_size, warmup_steps=5, num_steps=10):
    cmd = [
        "uv", "run", "python", "cs336_systems/end2end_benchmark.py",
        "--model_size", model_size,
        "--warmup_steps", str(warmup_steps),
        "--num_steps", str(num_steps)
    ]
    print(f"Running: {' '.join(cmd)} ...")
    try:
        result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='ignore', check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.stderr}"

def parse_results(output):
    res = {}
    # 匹配 Forward pass:  123.45 ms ± 0.12 ms
    fwd = re.search(r"Forward pass:\s+([\d\.]+)\s+ms\s+±\s+([\d\.]+)\s+ms", output)
    bwd = re.search(r"Backward pass:\s+([\d\.]+)\s+ms\s+±\s+([\d\.]+)\s+ms", output)
    if fwd: 
        res['fwd_avg'] = float(fwd.group(1))
        res['fwd_std'] = float(fwd.group(2))
    if bwd: 
        res['bwd_avg'] = float(bwd.group(1))
        res['bwd_std'] = float(bwd.group(2))
    return res

def main():
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write(f"Benchmark Run Started: {datetime.now()}\n" + "="*50 + "\n")

    print("\n--- Starting Base Benchmarks (Problem a & b) ---")
    base_data = []
    for size in MODEL_SIZES:
        output = run_benchmark(size)
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"\nModel: {size}\n{output}\n")
        metrics = parse_results(output)
        if metrics:
            metrics['model'] = size
            base_data.append(metrics)
        else:
            print(f"  -> Could not parse results for {size}. Check benchmark.log")

    print("\n--- Starting Warmup Experiments (Problem c) ---")
    warmup_data = []
    for w in WARMUP_EXPERIMENTS:
        output = run_benchmark("small", warmup_steps=w)
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"\nWarmup Experiment: {w} steps\n{output}\n")
        metrics = parse_results(output)
        if metrics:
            metrics['warmup'] = w
            warmup_data.append(metrics)

    print("\n" + "="*20 + " SUMMARY RESULTS " + "="*20)
    if base_data:
        df_base = pd.DataFrame(base_data)
        # 重新排列列顺序
        cols = ['model', 'fwd_avg', 'fwd_std', 'bwd_avg', 'bwd_std']
        print("\n### Table 1: Model Size Benchmarks (ms)")
        print(df_base[cols].to_string(index=False))

    if warmup_data:
        df_warmup = pd.DataFrame(warmup_data)
        cols_w = ['warmup', 'fwd_avg', 'fwd_std', 'bwd_avg', 'bwd_std']
        print("\n### Table 2: Warmup Impact (Small Model, ms)")
        print(df_warmup[cols_w].to_string(index=False))

    print(f"\nDetailed logs saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
