import torch
import argparse
import time
import numpy as np
from cs336_basics.model import BasicsTransformerLM
from jaxtyping import Float, Int
from torch import Tensor

# 移除全局变量，使用 MODEL_CONFIGS 里的配置
MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25,
    },
    "2.7B": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
}

def generate_data(batch_size: int, context_length: int, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """
    生成随机整数张量作为输入数据，模拟批处理的上下文长度。
    """
    input_data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    return input_data, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        choices=MODEL_CONFIGS.keys(),
        default="small",
        help="Model size to benchmark"
    )

    # 实验参数
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for benchmarking")
    parser.add_argument("--context_length", type=int, default=512, help="Context length for benchmarking")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size for the model")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")

    # 运行参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the benchmark on")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of steps to run the benchmark")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps before benchmarking")
    parser.add_argument("--mode", type=str, choices=["fwd", "fwd_bwd"], default="fwd_bwd", help="Benchmark mode: forward only or forward+backward")

    args = parser.parse_args()
    config = MODEL_CONFIGS[args.model_size]
    
    print(f"Running benchmark with config: Model={args.model_size}, Mode={args.mode}, Batch={args.batch_size}, SeqLen={args.context_length}, Device={args.device}")

    # 1. 初始化模型
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        **config
    ).to(args.device)

    model.train() # 即使只测前向，通常也保持 train 模式以模拟真实训练场景（除非明确为了推理测试）

    # 2. 准备数据
    input_ids, labels = generate_data(args.batch_size, args.context_length, args.vocab_size, args.device)

    # 3. 预热 (Warm-up) - 不记录时间
    print(f"Warming up for {args.warmup_steps} steps...")
    for _ in range(args.warmup_steps):
        logits = model(input_ids)
        if args.mode == "fwd_bwd":
            loss = logits.sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    # 4. 正式计时 (Benchmarking)
    print(f"Benchmarking for {args.num_steps} steps...")
    times = []
    timer = time.perf_counter 

    for _ in range(args.num_steps):
        torch.cuda.synchronize() # 确保上一步彻底结束
        start_time = timer()
        
        logits = model(input_ids)
        
        if args.mode == "fwd_bwd":
            loss = logits.sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
            
        torch.cuda.synchronize() # 等待这一步的所有 CUDA 核完成
        end_time = timer()
        times.append(end_time - start_time)

    # 5. 统计结果
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nResults ({args.mode}):")
    print(f"Average time per step: {avg_time*1000:.2f} ms")
    print(f"Standard deviation: {std_time*1000:.2f} ms")

if __name__ == "__main__":
    main()