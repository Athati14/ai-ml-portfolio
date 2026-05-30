"""Benchmark a transformer block under plain / AMP / FSDP regimes.

Launch FSDP run with:
  torchrun --nproc_per_node=2 bench_fsdp.py --mode fsdp
Single-process modes:
  python bench_fsdp.py --mode plain
  python bench_fsdp.py --mode amp
"""
from __future__ import annotations
import os, time, argparse, torch, torch.nn as nn


def build_model(dim=1024, layers=8):
    enc = nn.TransformerEncoderLayer(d_model=dim, nhead=8,
                                     dim_feedforward=4*dim, batch_first=True)
    return nn.TransformerEncoder(enc, num_layers=layers)


def batch(bs=16, seq=256, dim=1024, device="cuda"):
    return torch.randn(bs, seq, dim, device=device)


def run(mode: str, steps: int = 50):
    device = "cuda"
    model = build_model().to(device)

    if mode == "fsdp":
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        model = FSDP(model)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(mode == "amp"))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.time()

    for _ in range(steps):
        x = batch(device=device)
        opt.zero_grad()
        with torch.autocast("cuda", enabled=(mode == "amp")):
            loss = model(x).pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"mode={mode} steps={steps} time={dt:.2f}s "
          f"throughput={steps/dt:.2f} it/s peak_mem={peak:.2f}GB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["plain", "amp", "fsdp"], default="plain")
    run(p.parse_args().mode)
