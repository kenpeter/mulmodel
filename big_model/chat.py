"""
Chat REPL for BigModel (GPT-style causal language model).

Uses raw code prefixes — no system prompt. Just type a function signature
or partial code and the model continues it.

Usage:
  python -m big_model.chat                           # load best checkpoint
  python -m big_model.chat --checkpoint path/to.pt   # custom checkpoint
  python -m big_model.chat --temp 0.7 --top_k 40    # custom sampling
  python -m big_model.chat --tokens 200              # longer responses

Examples of good prompts:
  def is_prime(n):
  def binary_search(arr, target):
  for i in range(n):
  class Solution:
"""
from __future__ import annotations

import argparse
import sys

import torch

from big_model.transformer import BigModel

CHECKPOINT_DEFAULT = "big_model_data/big_model_best.pt"


def run_chat(
    checkpoint: str,
    temperature: float = 0.8,
    top_k: int = 50,
    max_new_tokens: int = 200,
    device: str | None = None,
) -> None:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [Chat] Loading model from {checkpoint}  (device={dev})")

    try:
        model = BigModel.load(checkpoint, device=dev)
    except FileNotFoundError:
        print(f"  [Chat] ERROR: checkpoint not found: {checkpoint}")
        print("  Run:  python -m big_model.pretrain --fresh  to train first.")
        sys.exit(1)

    model.eval()
    n = model.param_count()
    print(f"  [Chat] BigModel ready — {n:,} params")
    print(f"  [Chat] temperature={temperature}  top_k={top_k}  max_tokens={max_new_tokens}")
    print("─" * 60)
    print("Enter a code prefix (e.g. 'def fib(n):') and press Enter.")
    print("The model will complete it. 'quit' or Ctrl-C to exit.")
    print("─" * 60)

    while True:
        try:
            user_input = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Chat] Goodbye.")
            break

        if not user_input or user_input.lower() in {"quit", "exit", "q"}:
            print("[Chat] Goodbye.")
            break

        # Use the raw input as the prompt — no system prompt
        print(user_input, end="", flush=True)
        response = model.generate(
            user_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        print(response)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Complete code with BigModel")
    p.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT,
                   help=f"Path to model checkpoint (default: {CHECKPOINT_DEFAULT})")
    p.add_argument("--temp",    type=float, default=0.8,
                   help="Sampling temperature (default: 0.8)")
    p.add_argument("--top_k",  type=int,   default=50,
                   help="Top-k sampling cutoff (default: 50)")
    p.add_argument("--tokens", type=int,   default=200,
                   help="Max new tokens to generate (default: 200)")
    p.add_argument("--device", default=None,
                   help="Device: cuda / cpu (default: auto-detect)")
    args = p.parse_args()
    run_chat(
        checkpoint=args.checkpoint,
        temperature=args.temp,
        top_k=args.top_k,
        max_new_tokens=args.tokens,
        device=args.device,
    )
