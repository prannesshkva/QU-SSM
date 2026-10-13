"""
Benchmark 3: Inference Step Latency & Token Generation Throughput
Profiles per-token latency (ms/token) and throughput (tokens/second)
for QU-SSM's O(1) recurrent step compared to full sequence unroll.
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modeling_qu_ssm import QUSSMConfig, QUSSMForCausalLM

def run_throughput_benchmark():
    print("=" * 70)
    print(" BENCHMARK 3: GENERATION THROUGHPUT & STEP LATENCY ")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    config = QUSSMConfig(
        vocab_size=50257,
        d_model=512,
        d_state=8,
        n_layers=6,
        num_experts=8,
        moe_top_k=2
    )
    
    model = QUSSMForCausalLM(config).to(device)
    model.eval()
    
    batch_size = 1
    prompt_len = 64
    gen_tokens = 50
    warmup_iters = 3
    benchmark_iters = 10
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len), device=device)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Warming up ({warmup_iters} iterations)...")
    
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_ids)
            
        print(f"Benchmarking generation of {gen_tokens} tokens across {benchmark_iters} trials...")
        
        # Profile recurrent step execution
        latencies = []
        for _ in range(benchmark_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            # 1. Prefill prompt to initialize recurrent states
            states = [None] * len(model.layers)
            x_prompt = model.embed(input_ids)
            for t in range(prompt_len):
                x_t = x_prompt[:, t, :]
                for l_idx, layer in enumerate(model.layers):
                    x_t, states[l_idx] = layer.step(x_t, states[l_idx])
            
            # 2. Recurrent autoregressive generation
            curr_token = input_ids[:, -1:]
            for step_idx in range(gen_tokens):
                x = model.embed(curr_token)[:, 0, :]
                for l_idx, layer in enumerate(model.layers):
                    x, states[l_idx] = layer.step(x, states[l_idx])
                x = model.final_norm(x.unsqueeze(1))
                logits = model.lm_head(x)
                curr_token = torch.argmax(logits[:, -1:, :], dim=-1)
                
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            
    avg_total_time = sum(latencies) / len(latencies)
    tok_per_sec = gen_tokens / avg_total_time
    ms_per_tok = (avg_total_time / gen_tokens) * 1000.0
    
    print("-" * 70)
    print(f"Throughput:    {tok_per_sec:8.2f} tokens/second")
    print(f"Step Latency:  {ms_per_tok:8.2f} ms/token")
    print("=" * 70)
    
    return {
        "TokensPerSec": tok_per_sec,
        "StepLatencyMs": ms_per_tok
    }

if __name__ == "__main__":
    run_throughput_benchmark()
