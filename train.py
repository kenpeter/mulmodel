#!/usr/bin/env python3
"""
Train BigModel on codeforces_cots using byte-level tokenizer + SageAttention.

Usage:
    python train.py --epochs 10
    python train.py --epochs 10 --resume checkpoints/latest.pt
    python train.py --epochs 10 --warmup-steps 500 --lr 3e-4
"""

import argparse, os, time, glob
import torch
import torch.nn as nn
import pyarrow as pa
from torch.utils.data import Dataset, DataLoader
from transformer import BigModel, MODEL_CONFIG

# ── Dataset ───────────────────────────────────────────────────────────────────


class CodeforcesDataset(Dataset):
    def __init__(self, data_dir: str, context_length: int):
        arrow_files = sorted(glob.glob(f"{data_dir}/data-*.arrow"))
        texts = []
        for f in arrow_files:
            table = pa.ipc.open_stream(f).read_all()
            for i in range(len(table)):
                desc = table["description"][i].as_py() or ""
                gen = table["generation"][i].as_py() or ""
                texts.append(desc + "\n" + gen)

        all_bytes = bytearray(b"\x00".join(t.encode("utf-8") for t in texts))
        self.data = torch.frombuffer(all_bytes, dtype=torch.uint8).long()
        self.ctx = context_length
        print(
            f"[Data] {len(arrow_files)} shards, {len(texts):,} entries, "
            f"{len(self.data) / 1e9:.2f}B tokens, {len(self):,} samples"
        )

    def __len__(self):
        return (len(self.data) - self.ctx) // self.ctx

    def __getitem__(self, i):
        start = i * self.ctx
        chunk = self.data[start : start + self.ctx + 1]
        return chunk[:-1], chunk[1:]


# ── LR Schedule: linear warmup → cosine decay ────────────────────────────────


def get_lr(step: int, warmup_steps: int, total_steps: int, lr: float) -> float:
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))


# ── Training ──────────────────────────────────────────────────────────────────


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = BigModel(MODEL_CONFIG).to(device=device, dtype=dtype)
    print(f"[BigModel] {model.num_params():,} params  {dtype} on {device}")

    dataset = CodeforcesDataset(
        "data/code/codeforces_cots", MODEL_CONFIG["context_length"]
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck.get("epoch", 0) + 1
        global_step = ck.get("step", 0)
        best_loss = ck.get("best_loss", float("inf"))
        print(
            f"[Resume] epoch {start_epoch}, step {global_step}, best_loss {best_loss:.4f}"
        )

    total_steps = args.epochs * len(loader)
    print(
        f"[Schedule] warmup {args.warmup_steps} steps, total {total_steps} steps, lr {args.lr}"
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    loss_fn = nn.CrossEntropyLoss()
    train_start = time.time()
    training_seconds = 0.0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, steps = 0.0, 0
        accum_loss = 0.0
        epoch_start = time.time()

        done = False
        for x, y in loader:
            if args.time_limit and (time.time() - train_start) >= args.time_limit:
                done = True
                break

            # LR warmup / cosine decay
            lr = get_lr(global_step, args.warmup_steps, total_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss = loss / args.grad_accum  # normalize loss for accumulation

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1

            total_loss += loss.item() * args.grad_accum
            steps += 1

            if global_step % args.log_every == 0:
                avg = total_loss / steps
                elapsed = time.time() - train_start
                print(
                    f"  step {global_step}  loss {avg:.4f}  lr {lr:.2e}  elapsed {elapsed:.0f}s",
                    flush=True,
                )

        training_seconds += time.time() - epoch_start
        if done:
            break
        avg_loss = total_loss / max(steps, 1)
        lr_now = get_lr(global_step, args.warmup_steps, total_steps, args.lr)
        print(
            f"epoch {epoch + 1}/{args.epochs}  loss {avg_loss:.4f}  lr {lr_now:.2e}  steps {global_step}"
        )

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        if (epoch + 1) % args.save_every == 0 or is_best:
            ck = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "loss": avg_loss,
                "best_loss": best_loss,
            }
            torch.save(ck, os.path.join(args.checkpoint_dir, "latest.pt"))
            if is_best:
                torch.save(ck, os.path.join(args.checkpoint_dir, "best.pt"))
                print(f"  [best] {best_loss:.4f}")

    peak_vram = (
        torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    )
    print("---")
    print(f"train_loss:       {best_loss:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Linear warmup steps before cosine decay",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
    )
    p.add_argument(
        "--log-every", type=int, default=3000, help="Print loss every N steps"
    )
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        nargs="?",
        const="checkpoints/latest.pt",
        help="Resume from checkpoint (default: checkpoints/latest.pt)",
    )
    p.add_argument("--time-limit", type=int, default=None)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
