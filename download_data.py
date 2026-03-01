"""
Download training data for BigModel pretraining.

Downloads:
  1. Wikitext-103 (~520MB)  → data/wikitext        (clean Wikipedia text)
  2. GSM8K        (~8MB)    → data/math/gsm8k       (grade school math problems)
  3. MATH dataset (~90MB)   → data/math/competition  (competition math problems)

Resumable: re-running skips already-downloaded datasets.

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

TEXT_DATASET   = "wikitext"
TEXT_CONFIG    = "wikitext-103-v1"      # 500MB, no auth needed, clean Wikipedia text
TEXT_SPLIT     = "train"
TEXT_SAVE_DIR  = "data/wikitext"

# Math datasets (both work without loading scripts)
MATH_SAVE_DIR  = "data/math"
MATH_SOURCES = [
    # (dataset_id, config, split, save_subdir, label)
    ("openai/gsm8k",               "main", "train", "gsm8k",       "GSM8K ~8MB  grade school math"),
    ("hendrycks/competition_math", None,   "train", "competition",  "MATH  ~90MB competition math"),
]


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

    ds = load_dataset(TEXT_DATASET, TEXT_CONFIG, split=TEXT_SPLIT)

    print(f"  Downloaded {len(ds):,} articles in {time.time()-t0:.0f}s")
    print(f"  Saving to {TEXT_SAVE_DIR} ...")

    os.makedirs(TEXT_SAVE_DIR, exist_ok=True)
    ds.save_to_disk(TEXT_SAVE_DIR)

    size = _dir_size(TEXT_SAVE_DIR)
    print(f"  Done. Size on disk: {_hr(size)}")


# ── math download ─────────────────────────────────────────────────────────────

def download_math() -> None:
    print("\n" + "=" * 60)
    print("MATH: GSM8K + Competition MATH  (~100MB total)")
    print("=" * 60)

    from datasets import load_dataset

    for dataset_id, config, split, subdir, label in MATH_SOURCES:
        save_dir = os.path.join(MATH_SAVE_DIR, subdir)

        if _already_saved(save_dir):
            print(f"  [skip]  {label}  already saved")
            continue

        print(f"  [down]  {label}  downloading...")
        t0 = time.time()
        try:
            ds = load_dataset(dataset_id, config, split=split) if config \
                 else load_dataset(dataset_id, split=split)
            os.makedirs(save_dir, exist_ok=True)
            ds.save_to_disk(save_dir)
            print(f"          {len(ds):>8,} examples  {time.time()-t0:.0f}s  ✓")
        except Exception as e:
            print(f"          ERROR: {e} — skipping")

    size = _dir_size(MATH_SAVE_DIR)
    print(f"\n  Math data total: {_hr(size)}")


# ── summary ───────────────────────────────────────────────────────────────────

def summary() -> None:
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for label, path in [("Wikitext-103", TEXT_SAVE_DIR), ("Math (GSM8K+MATH)", MATH_SAVE_DIR)]:
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
