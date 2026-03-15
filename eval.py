#!/usr/bin/env python3
"""
Evaluate BigModel: perplexity + interactive Codeforces chat mode.

Usage:
    python eval.py                          # perplexity eval
    python eval.py --chat                   # interactive problem → solution chat
    python eval.py --checkpoint checkpoints/latest.pt --chat
"""

import argparse, math, os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer import BigModel, MODEL_CONFIG
from train import CodeforcesDataset


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, prompt: str, max_new_tokens: int, device, dtype,
             temperature=0.8, top_k=50):
    model.eval()
    ctx_len = MODEL_CONFIG["context_length"]
    ids = torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long, device=device)
    ids = ids[-ctx_len:].unsqueeze(0)  # [1, T]

    for _ in range(max_new_tokens):
        inp = ids[:, -ctx_len:]
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(inp)
        logits = logits[0, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)

    out_ids = ids[0, -max_new_tokens:].tolist()
    return bytes(out_ids).decode("utf-8", errors="replace")


# ── Perplexity ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, loader, device, dtype, max_batches=200):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
        n += 1
    avg_loss = total_loss / max(n, 1)
    return avg_loss, math.exp(avg_loss)


# ── Chat mode ─────────────────────────────────────────────────────────────────

def build_prompt(problem: str) -> str:
    """Wrap a problem statement in the training-data format the model learned."""
    return problem.strip() + "\n"


def chat_loop(model, device, dtype, args):
    print("\n" + "=" * 60)
    print("  BigModel — Codeforces Chat")
    print("  Paste a problem statement, then press Enter twice to submit.")
    print("  Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        print("Problem> ", end="", flush=True)
        lines = []
        try:
            while True:
                line = input()
                if line.lower() in ("quit", "exit"):
                    print("Bye.")
                    return
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
        except EOFError:
            return

        problem = "\n".join(lines).strip()
        if not problem:
            continue

        prompt = build_prompt(problem)
        print(f"\n[Generating up to {args.max_new_tokens} tokens...]\n")

        out = generate(model, prompt, args.max_new_tokens, device, dtype,
                       args.temperature, args.top_k)

        print("-" * 60)
        print(problem)
        print(out)
        print("-" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--chat", action="store_true", help="Interactive problem→solution mode")
    p.add_argument("--data-dir", default="data/code/codeforces_cots")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=200)
    p.add_argument("--val-split", type=float, default=0.05)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[Error] Checkpoint not found: {args.checkpoint}")
        print("  Run 'python train.py --epochs 10' first, then re-run eval.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load model
    model = BigModel(MODEL_CONFIG).to(device=device, dtype=dtype)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"])
    print(f"[Loaded] {args.checkpoint}")
    print(f"  epoch={ck.get('epoch','?')}  step={ck.get('step','?')}  "
          f"train_loss={ck.get('loss', float('nan')):.4f}")
    print(f"  params={model.num_params():,}  dtype={dtype}  device={device}")

    if args.chat:
        chat_loop(model, device, dtype, args)
        return

    # ── Perplexity eval ───────────────────────────────────────────────────────
    dataset = CodeforcesDataset(args.data_dir, MODEL_CONFIG["context_length"])
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    _, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"\n[Eval] val_samples={n_val}  max_batches={args.max_batches}")
    avg_loss, ppl = compute_perplexity(model, val_loader, device, dtype, args.max_batches)
    print(f"  val_loss={avg_loss:.4f}  perplexity={ppl:.2f}")

    # Quick generation samples
    samples = [
        "Given an array of N integers, find the maximum subarray sum.\n",
        "def solve(n, arr):\n    # find pairs summing to target\n",
        "You are given a tree with N nodes. Find the diameter.\n",
    ]
    print(f"\n[Samples] max_new_tokens={args.max_new_tokens}  temp={args.temperature}  top_k={args.top_k}")
    for prompt in samples:
        out = generate(model, prompt, args.max_new_tokens, device, dtype,
                       args.temperature, args.top_k)
        print(f"\n{prompt}{out}")
        print("-" * 60)


if __name__ == "__main__":
    main()
