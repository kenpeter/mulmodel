#!/usr/bin/env python3
"""
Start auto-research for the mulmodel project.

Usage:
    python start.py

    python start.py --inspect

    python start.py --continuous  # keep auto-restarting

    python start.py --serve      # use persistent server mode (recommended for stability)

The script uses 'opencode run' to execute the auto-research loop.
It reads program.md and issues the autoresearch command.

Connection stability tips:
- Use --serve for a persistent server (recommended for long sessions)
- Use --continuous to auto-restart if opencode crashes/disconnects
- Each 'opencode run' call is independent; no shared state between calls
- If you get connection errors, try: opencode auth login
"""

import subprocess
import sys
import argparse
import os
import time
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, ".opencode_config.json")

AUTORESEARCH_MSG = """read program.md auto research"""

DEFAULT_TIMEOUT = 600


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def run_opencode_run(message: str, timeout: int = 600, retries: int = 3) -> int:
    """Run opencode with a message. Returns exit code."""
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["opencode", "run", message],
                cwd=SCRIPT_DIR,
                timeout=timeout,
            )
            return result.returncode
        except subprocess.TimeoutExpired:
            print(f"[start.py] opencode timed out (attempt {attempt + 1}/{retries})")
            if attempt < retries - 1:
                time.sleep(5)
        except FileNotFoundError:
            print(
                "[start.py] ERROR: 'opencode' command not found. Is opencode installed?"
            )
            return 1
        except Exception as e:
            print(f"[start.py] opencode error: {e}")
            if attempt < retries - 1:
                time.sleep(5)
    print("[start.py] All retries failed.")
    return 1


def run_opencode_serve(message: str, timeout: int = 600, retries: int = 3) -> int:
    """Start server, send message, return exit code. More stable for long sessions."""
    server_proc = None
    for attempt in range(retries):
        try:
            server_proc = subprocess.Popen(
                ["opencode", "serve"],
                cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(8)

            result = subprocess.run(
                ["opencode", "run", message],
                cwd=SCRIPT_DIR,
                timeout=timeout,
            )
            return result.returncode
        except subprocess.TimeoutExpired:
            print(
                f"[start.py] opencode serve timed out (attempt {attempt + 1}/{retries})"
            )
            if attempt < retries - 1:
                time.sleep(5)
        except FileNotFoundError:
            print("[start.py] ERROR: 'opencode' command not found.")
            return 1
        except Exception as e:
            print(f"[start.py] opencode serve error: {e}")
            if attempt < retries - 1:
                time.sleep(5)
        finally:
            if server_proc:
                server_proc.terminate()
                server_proc.wait(timeout=5)
    print("[start.py] All retries failed.")
    return 1


def main():
    p = argparse.ArgumentParser(description="Start mulmodel auto-research")
    p.add_argument("--inspect", action="store_true", help="Inspect mode")
    p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout for opencode run (default {DEFAULT_TIMEOUT}s)",
    )
    p.add_argument(
        "--continuous",
        action="store_true",
        help="Keep running in a loop (auto-restart on failure)",
    )
    p.add_argument(
        "--serve",
        action="store_true",
        help="Use persistent server mode (more stable for long sessions)",
    )
    args = p.parse_args()

    mode = "inspect" if args.inspect else "auto"
    runner = run_opencode_serve if args.serve else run_opencode_run

    if args.continuous:
        print("[start.py] Continuous mode: will auto-restart on failure")
        failures = 0
        while True:
            print(f"\n{'=' * 60}")
            print(f"[start.py] === Cycle {failures + 1} ===")
            print(f"{'=' * 60}")
            ret = runner(AUTORESEARCH_MSG, timeout=args.timeout)
            failures += 1
            if ret == 0:
                print(f"[start.py] opencode finished successfully (cycle #{failures}).")
                break
            print(
                f"[start.py] Cycle #{failures} failed (exit {ret}), restart in 15s..."
            )
            time.sleep(15)
    else:
        ret = runner(AUTORESEARCH_MSG, timeout=args.timeout)
        sys.exit(ret)


if __name__ == "__main__":
    main()
