"""
Download coding datasets for BigModel pretraining.

Downloads:
  1. Nan-Do/leetcode_contests    (~2GB)  → data/code/leetcode         (4.7M submissions, 2406 problems)
  2. deepmind/code_contests      (~3GB)  → data/code/code_contests    (11k problems + 13M solutions)
  3. codeparrot/apps             (~1GB)  → data/code/apps             (10k problems with test cases)
  4. open-r1/codeforces-cots     (~500MB)→ data/code/codeforces_cots  (Codeforces + chain-of-thought)

Resume: HuggingFace caches shards in ~/.cache/huggingface/datasets/
        Re-running resumes automatically from last complete shard.
        Already-saved datasets are skipped entirely.

Usage:
  python download_data.py               # download all
  python download_data.py --leetcode    # LeetCode only
  python download_data.py --contests    # DeepMind CodeContests only
  python download_data.py --apps        # APPS only
  python download_data.py --codeforces  # Codeforces-CoTs only
"""
from __future__ import annotations

import argparse
import os
import time


# ── dataset configs ────────────────────────────────────────────────────────────

SAVE_BASE = "data/code"

DATASETS = [
    {
        "flag":       "leetcode",
        "id":         "Nan-Do/leetcode_contests",
        "config":     None,
        "split":      "train",
        "save_dir":   "leetcode",
        "label":      "LeetCode contests",
        "approx_size":"~2GB",
        "n_approx":   "4.7M submissions",
    },
    {
        "flag":       "contests",
        "id":         "deepmind/code_contests",
        "config":     None,
        "split":      "train",
        "save_dir":   "code_contests",
        "label":      "DeepMind CodeContests",
        "approx_size":"~3GB",
        "n_approx":   "11k problems + 13M solutions",
    },
    {
        "flag":       "apps",
        "id":         "codeparrot/apps",
        "config":     None,
        "split":      "train",
        "save_dir":   "apps",
        "label":      "APPS",
        "approx_size":"~1GB",
        "n_approx":   "10k problems with test cases",
    },
    {
        "flag":       "codeforces",
        "id":         "open-r1/codeforces-cots",
        "config":     None,
        "split":      "train",
        "save_dir":   "codeforces_cots",
        "label":      "Codeforces-CoTs",
        "approx_size":"~500MB",
        "n_approx":   "few k problems with chain-of-thought",
    },
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _hr(n_bytes: int) -> str:
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
    """dataset_info.json is written by save_to_disk() on success."""
    return os.path.exists(os.path.join(path, "dataset_info.json"))


# ── download ───────────────────────────────────────────────────────────────────

def download_one(ds_cfg: dict) -> None:
    save_dir = os.path.join(SAVE_BASE, ds_cfg["save_dir"])

    if _already_saved(save_dir):
        size = _dir_size(save_dir)
        print(f"  [skip]  {ds_cfg['label']:<25}  already saved  ({_hr(size)})")
        return

    from datasets import load_dataset

    print(f"\n  [down]  {ds_cfg['label']}")
    print(f"          {ds_cfg['n_approx']}  {ds_cfg['approx_size']}")
    print(f"          HF cache resumes if interrupted — just re-run")
    t0 = time.time()

    try:
        if ds_cfg["config"]:
            ds = load_dataset(ds_cfg["id"], ds_cfg["config"], split=ds_cfg["split"])
        else:
            ds = load_dataset(ds_cfg["id"], split=ds_cfg["split"])

        os.makedirs(save_dir, exist_ok=True)
        print(f"          Saving {len(ds):,} rows to {save_dir} ...")
        ds.save_to_disk(save_dir)

        size = _dir_size(save_dir)
        elapsed = time.time() - t0
        print(f"          Done  {len(ds):,} rows  {_hr(size)}  {elapsed:.0f}s  ✓")

    except KeyboardInterrupt:
        print(f"\n  [interrupted]  {ds_cfg['label']} — HF cache preserved, re-run to resume")
        raise
    except Exception as e:
        print(f"  [error]   {ds_cfg['label']}: {e}")


# ── summary ────────────────────────────────────────────────────────────────────

def summary() -> None:
    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    total = 0
    for ds_cfg in DATASETS:
        save_dir = os.path.join(SAVE_BASE, ds_cfg["save_dir"])
        if _already_saved(save_dir):
            size = _dir_size(save_dir)
            total += size
            print(f"  ✓  {ds_cfg['label']:<25}  {save_dir:<30}  {_hr(size):>8}")
        else:
            print(f"  ✗  {ds_cfg['label']:<25}  {save_dir:<30}  {'not downloaded':>8}")
    print(f"\n  Total on disk: {_hr(total)}")
    print("\n  Next step:")
    print("    python -m big_model.pretrain --epochs 20 --steps 200")
    print("=" * 65)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download coding datasets")
    parser.add_argument("--leetcode",   action="store_true", help="LeetCode contests only")
    parser.add_argument("--contests",   action="store_true", help="DeepMind CodeContests only")
    parser.add_argument("--apps",       action="store_true", help="APPS only")
    parser.add_argument("--codeforces", action="store_true", help="Codeforces-CoTs only")
    args = parser.parse_args()

    flags = {
        "leetcode":   args.leetcode,
        "contests":   args.contests,
        "apps":       args.apps,
        "codeforces": args.codeforces,
    }
    # default: download all
    download_all = not any(flags.values())

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("ERROR: 'datasets' not installed.  Run: pip install datasets")
        return

    print("=" * 65)
    print("Coding dataset downloader  (re-run at any time to resume)")
    print("=" * 65)

    for ds_cfg in DATASETS:
        if download_all or flags.get(ds_cfg["flag"]):
            download_one(ds_cfg)

    summary()


if __name__ == "__main__":
    main()
