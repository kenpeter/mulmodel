"""
Download training data for BigModel pretraining.

Downloads:
  1. OpenWebText 13% (~5GB)  → data/openwebtext
  2. DeepMind Math (~2GB)    → data/deepmind_math/<module>/

Resumable: re-running skips already-downloaded shards (HuggingFace cache).
For true mid-file resume, use: huggingface-cli download (see bottom of script).

Usage:
  python download_data.py             # download both
  python download_data.py --text      # text only
  python download_data.py --math      # math only
"""
from __future__ import annotations

import argparse
import os
import time


# ── config ────────────────────────────────────────────────────────────────────

TEXT_DATASET   = "openwebtext"
TEXT_SPLIT     = "train[:13%]"          # ~5GB
TEXT_SAVE_DIR  = "data/openwebtext"

# DeepMind math modules that match what this system does:
#   sequences, arithmetic, algebra — not calculus/probability (too advanced)
MATH_MODULES = [
    "arithmetic__add_or_sub",
    "arithmetic__add_or_sub_multiple",
    "arithmetic__mul",
    "arithmetic__div",
    "arithmetic__mixed",
    "algebra__linear_1d",
    "algebra__linear_2d",
    "numbers__place_value",
    "numbers__list_prime_factors",
    "numbers__gcd",
    "numbers__lcm",
]
MATH_SAVE_DIR  = "data/deepmind_math"


# ── helpers ───────────────────────────────────────────────────────────────────

def _hr(n_bytes: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _already_saved(path: str) -> bool:
    """Check if dataset_info.json exists (save_to_disk marker)."""
    return os.path.exists(os.path.join(path, "dataset_info.json"))


# ── text download ─────────────────────────────────────────────────────────────

def download_text() -> None:
    print("\n" + "=" * 60)
    print("TEXT: OpenWebText 13%  (~5GB)")
    print("=" * 60)

    if _already_saved(TEXT_SAVE_DIR):
        size = _dir_size(TEXT_SAVE_DIR)
        print(f"  Already downloaded → {TEXT_SAVE_DIR}  ({_hr(size)})")
        return

    from datasets import load_dataset

    print(f"  Downloading {TEXT_DATASET} split={TEXT_SPLIT} ...")
    print("  (HuggingFace cache: re-run resumes automatically)")
    t0 = time.time()

    ds = load_dataset(TEXT_DATASET, split=TEXT_SPLIT)

    print(f"  Downloaded {len(ds):,} articles in {time.time()-t0:.0f}s")
    print(f"  Saving to {TEXT_SAVE_DIR} ...")

    os.makedirs(TEXT_SAVE_DIR, exist_ok=True)
    ds.save_to_disk(TEXT_SAVE_DIR)

    size = _dir_size(TEXT_SAVE_DIR)
    print(f"  Done. Size on disk: {_hr(size)}")


# ── math download ─────────────────────────────────────────────────────────────

def download_math() -> None:
    print("\n" + "=" * 60)
    print("MATH: DeepMind Mathematics Dataset  (~2GB total)")
    print("=" * 60)

    from datasets import load_dataset

    total_examples = 0
    for module in MATH_MODULES:
        save_dir = os.path.join(MATH_SAVE_DIR, module)

        if _already_saved(save_dir):
            print(f"  [skip]  {module:<45}  already saved")
            continue

        print(f"  [down]  {module:<45}  downloading...")
        t0 = time.time()

        try:
            ds = load_dataset(
                "deepmind/math_dataset",
                module,
                split="train",
            )
            os.makedirs(save_dir, exist_ok=True)
            ds.save_to_disk(save_dir)
            total_examples += len(ds)
            print(f"          {len(ds):>8,} examples  {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"          ERROR: {e} — skipping")

    size = _dir_size(MATH_SAVE_DIR)
    print(f"\n  Math data total: {_hr(size)}")


# ── summary ───────────────────────────────────────────────────────────────────

def summary() -> None:
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for label, path in [("OpenWebText", TEXT_SAVE_DIR), ("DeepMind Math", MATH_SAVE_DIR)]:
        if os.path.exists(path):
            size = _dir_size(path)
            status = "ready" if _already_saved(path) or os.path.isdir(path) else "incomplete"
            print(f"  {label:<20} {path:<30} {_hr(size):>10}  [{status}]")
        else:
            print(f"  {label:<20} {path:<30} {'not downloaded':>10}")

    print()
    print("  Next step:")
    print("    python -m big_model.pretrain --epochs 10 --steps 200")
    print("=" * 60)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download BigModel training data")
    parser.add_argument("--text", action="store_true", help="Download text only")
    parser.add_argument("--math", action="store_true", help="Download math only")
    args = parser.parse_args()

    # default: download both
    do_text = args.text or (not args.text and not args.math)
    do_math = args.math or (not args.text and not args.math)

    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        print("ERROR: 'datasets' not installed.")
        print("  Run:  pip install datasets")
        return

    if do_text:
        download_text()
    if do_math:
        download_math()

    summary()


if __name__ == "__main__":
    main()
