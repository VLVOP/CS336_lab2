import argparse
import torch.cuda.nvtx as nvtx
import torch
from torch import einsum
from math import sqrt
from jaxtyping import Float
from torch import Tensor
import cs336_basics.model
from cs336_basics.nn_utils import softmax
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

def annotate_scaled_dot_product(Q, K, V, mask=None):
    with nvtx.range("scaled_dot_product_attention"):
        d_k = K.shape[-1]

        with nvtx.range("computing attention scores"):
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / sqrt(d_k)

            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with nvtx.range("computing softmax"):
            attention_weights = softmax(attention_scores, dim=-1)

        with nvtx.range("final matmul"):
            return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")
        
print("Monkey-patching cs336_basics.model.scaled_dot_product_attention with annotated...")
cs336_basics.model.scaled_dot_product_attention = annotate_scaled_dot_product


MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def generate_data(batch_size: int, context_length: int, vocab_size: int, device: torch.device) -> Tensor["batch context_length ..."]:
    """
    生成随机整数张量作为输入数据，模拟批处理的上下文长度。
    """
    input_data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    return input_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, choices=MODEL_CONFIGS.keys(), default="small")
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--active_steps", type=int, default=5)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        print("CUDA is not available. Exiting.")
        device = torch.device("cpu")
    
    config = MODEL_CONFIGS[args.model_size]

    print(f"Profiling config: {args.model_size}, Context: {args.context_length}, Batch: {args.batch_size}")

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=10000.0,
        **config
    ).to(device)

    model.train()

    optimizer = AdamW(model.parameters())

    input_ids = generate_data(args.batch_size, args.context_length, args.vocab_size, device)

    print("Starting warmup...")
    with nvtx.range("warmup"):
        for _ in range(args.warmup_steps):
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = logits.sum()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

    print(f"Runing {args.active_steps} profiled steps...")

    with nvtx.range("profiling_loop"):
        for step in range(args.active_steps):

            with nvtx.range(f"step_{step}"):
                optimizer.zero_grad(set_to_none=True)
                
                with nvtx.range("forward"):
                    logits = model(input_ids)
                    loss = logits.sum()

                with nvtx.range("backward"):
                    loss.backward()

                with nvtx.range("optimizer"):
                    optimizer.step()

                torch.cuda.synchronize()

    print("Profiling complete.")

if __name__ == "__main__":
    main()