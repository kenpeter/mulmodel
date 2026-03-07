"""Quick coding eval for BigModel."""
import math
import torch
from big_model.transformer import BigModel, MAX_SEQ

CHECKPOINT = "big_model_data/big_model_best.pt"

SHORT_PROMPTS = [
    "def reverse(s):",
    "def is_prime(n):",
    "def fib(n):",
    "def bubble_sort(arr):",
    "for i in range(n):\n    ",
]


def measure_loss(model, text: str, device: str) -> float:
    """Cross-entropy loss on a text snippet in bits-per-byte."""
    b = list(text.encode("utf-8", errors="replace"))[:MAX_SEQ]
    if len(b) < 2:
        return float("nan")
    ids = torch.tensor([b], dtype=torch.long, device=device)
    with torch.no_grad():
        loss = model.pretrain_causal(ids)
    return loss.item() / math.log(2)  # nats → bits


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {CHECKPOINT} on {dev}...")
    model = BigModel.load(CHECKPOINT, device=dev)
    model = model.to(dev)  # ensure model is fully on device
    model.eval()
    print(f"Params: {model.param_count():,}")
    print("=" * 70)

    code_snippets = [
        "def reverse(s):\n    return s[::-1]",
        "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "for i in range(n):\n    print(i)",
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a+b\n    return a",
    ]
    random_snippets = [
        "xkqz wlmpr jvbt oqfz ncrp xlm wbt",
        "zzz aaa bbb ccc ddd eee fff ggg hhh",
        "1234567890 abcdefghijklmnopqrstuvwxyz @@@@",
        "The quick brown fox jumps over the lazy dog.",
    ]

    print("\n[A] Bits-per-byte (lower = model assigns higher probability):")
    print("  Code snippets:")
    code_bpbs = []
    for s in code_snippets:
        bpb = measure_loss(model, s, dev)
        code_bpbs.append(bpb)
        print(f"    {bpb:.3f} bpb  |  {s[:55]!r}")
    print(f"  >>> Avg code bpb: {sum(code_bpbs)/len(code_bpbs):.3f}")

    print("  Random text:")
    rand_bpbs = []
    for s in random_snippets:
        bpb = measure_loss(model, s, dev)
        rand_bpbs.append(bpb)
        print(f"    {bpb:.3f} bpb  |  {s[:55]!r}")
    print(f"  >>> Avg random bpb: {sum(rand_bpbs)/len(rand_bpbs):.3f}")

    code_avg = sum(code_bpbs) / len(code_bpbs)
    rand_avg = sum(rand_bpbs) / len(rand_bpbs)
    print(f"\n  Code vs Random gap: {rand_avg - code_avg:+.3f} bpb")
    print(f"  (positive = model assigns higher prob to code than random noise)")

    print("\n[B] Generation (temp=0.7, top_k=40, max_new=50):")
    for p in SHORT_PROMPTS:
        out = model.generate(p, max_new_tokens=50, temperature=0.7, top_k=40)
        print(f"  PROMPT: {p!r}")
        print(f"  OUTPUT: {out!r}")
        print()


if __name__ == "__main__":
    main()
