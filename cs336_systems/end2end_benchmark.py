import torch
import argparse
from cs336_basics.model import BasicsTransformerLM
from jaxtyping import Float, Int
from torch import Tensor

vocab_size = 10000
batch_size = 4
context_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def generate_data(batch_size: int, context_length: int, device: torch.device) -> Int[Tensor, "batch context_len"]:
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
    parser.add_argument("--batch_size", type=int, default=batch_size, help="Batch size for benchmarking")
    parser.add_argument("--context_length", type=int, default=context_length, help="Context length for benchmarking")
    parser.add_argument("--vocab_size", type=int, default=vocab_size, help="Vocabulary size for the model")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")

    # 运行参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the benchmark on")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of steps to run the benchmark")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps before benchmarking")
    parser.add_argument("--mode", type=str, choices=["fwd", "fwd_bwd"], default="fwd_bwd", help="Benchmark mode: forward only or forward+backward")

    args = parser.parse_args()
    config = MODEL_CONFIGS[args.model_size]

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        **config
    ).to(args.device)


    model.train()

    input_ids, labels = generate_data(args.batch_size, args.context_length, args.device)

    print(f"Warming up for {args.warmup_steps} steps...")
    for _ in range(args.warmup_steps):
        
        logits = model(input_ids)
        if args.mode == "fwd_bwd":

            loss = logits.sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    print(f"Benchmarking for {args.num_steps} steps...")