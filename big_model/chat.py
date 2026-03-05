"""
Chat REPL for BigModel (GPT-style causal language model).

Usage:
  python -m big_model.chat                           # load best checkpoint
  python -m big_model.chat --checkpoint path/to.pt   # custom checkpoint
  python -m big_model.chat --temp 0.7 --top_k 40    # custom sampling
  python -m big_model.chat --tokens 512              # longer responses

Type your coding question and press Enter.
Type 'quit', 'exit', or press Ctrl-C to stop.
"""
from __future__ import annotations

import argparse
import sys

import torch

from big_model.transformer import BigModel

CHECKPOINT_DEFAULT = "big_model_data/big_model_best.pt"
SYSTEM_PROMPT = (
    "You are a helpful assistant that solves competitive programming problems. "
    "Provide clear, efficient solutions with explanations.\n\n"
)


def run_chat(
    checkpoint: str,
    temperature: float = 0.8,
    top_k: int = 50,
    max_new_tokens: int = 256,
    device: str | None = None,
) -> None:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [Chat] Loading model from {checkpoint}  (device={dev})")

    try:
        model = BigModel.load(checkpoint, device=dev)
    except FileNotFoundError:
        print(f"  [Chat] ERROR: checkpoint not found: {checkpoint}")
        print("  Run:  python -m big_model.pretrain  to train first.")
        sys.exit(1)

    model.eval()
    n = model.param_count()
    print(f"  [Chat] BigModel ready — {n:,} params")
    print(f"  [Chat] temperature={temperature}  top_k={top_k}  max_tokens={max_new_tokens}")
    print("─" * 60)
    print("Type a coding question. 'quit' or Ctrl-C to exit.")
    print("─" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Chat] Goodbye.")
            break

        if not user_input or user_input.lower() in {"quit", "exit", "q"}:
            print("[Chat] Goodbye.")
            break

        prompt = SYSTEM_PROMPT + "Question: " + user_input + "\n\nAnswer:\n"

        print("\nAssistant: ", end="", flush=True)
        response = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        print(response)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Chat with BigModel (GPT-style causal LM)")
    p.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT,
                   help=f"Path to model checkpoint (default: {CHECKPOINT_DEFAULT})")
    p.add_argument("--temp",    type=float, default=0.8,
                   help="Sampling temperature (default: 0.8)")
    p.add_argument("--top_k",  type=int,   default=50,
                   help="Top-k sampling cutoff (default: 50)")
    p.add_argument("--tokens", type=int,   default=256,
                   help="Max new tokens to generate (default: 256)")
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
