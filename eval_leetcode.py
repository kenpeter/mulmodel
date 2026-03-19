#!/usr/bin/env python3
"""
LeetCode evaluation - use actual problems from newfacade_LeetCodeDataset.
Generates Python solutions and runs against test cases.
"""

import json
import subprocess
import tempfile
import os
import re
import sys

import torch

from transformer import BigModel, MODEL_CONFIG


def load_test_problems(data_dir: str, max_problems: int = 10):
    """Load problems from leetcode_test.jsonl"""
    problems = []
    test_file = os.path.join(data_dir, "leetcode_test.jsonl")
    with open(test_file, "r") as f:
        for line in f:
            item = json.loads(line)
            problems.append(item)
            if len(problems) >= max_problems:
                break
    return problems


def generate(
    model,
    prompt: str,
    max_tokens: int,
    device,
    dtype,
    temperature=0.8,
    top_k=50,
    tokenizer=None,
):
    model.eval()
    if tokenizer is not None:
        ids = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.long, device=device
        ).unsqueeze(0)
    else:
        ids = torch.tensor(
            list(prompt.encode("utf-8")), dtype=torch.long, device=device
        ).unsqueeze(0)
    for _ in range(max_tokens):
        inp = ids[:, -MODEL_CONFIG["context_length"] :]
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(inp)
        logits = logits[0, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
        probs = torch.softmax(logits, dim=0)
        next_id = torch.multinomial(probs, 1).unsqueeze(0)
        ids = torch.cat([ids, next_id], dim=1)
    if tokenizer is not None:
        return tokenizer.decode(ids[0].tolist())
    return bytes(ids[0].tolist()).decode("utf-8", errors="replace")


def extract_python_code(output: str) -> str:
    if "<|endoftext|>" in output:
        output = output.split("<|endoftext|>")[0]

    match = re.search(r"```python\s*\n(.*?)```", output, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n(.*?)```", output, re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = output.split("\n")
    for i, line in enumerate(lines):
        if "class Solution" in line or "def " in line:
            result = []
            for l in lines[i:]:
                if l.strip().startswith("[CODE]") or l.strip().startswith("```"):
                    break
                result.append(l)
            if result:
                return "\n".join(result).strip()

    return ""


def run_python_code(code: str, test_code: str) -> tuple[bool, str]:
    """Run solution code with test harness. Returns (passed, output)."""
    if not code.strip():
        return False, "NO_CODE"

    full_code = code + "\n\n" + test_code

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(full_code)
        py_file = f.name

    try:
        result = subprocess.run(
            [sys.executable, py_file],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return False, f"RUNTIME_ERROR: {result.stderr[:300]}"
        return True, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"ERROR: {e}"
    finally:
        if os.path.exists(py_file):
            os.unlink(py_file)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--ctx-len", type=int, default=512)
    p.add_argument(
        "--tokenizer", type=str, default="byte", choices=["byte", "tiktoken"]
    )
    p.add_argument("--num-problems", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=1024)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = None
    if args.tokenizer == "tiktoken":
        import tiktoken

        MODEL_CONFIG["vocab_size"] = 50257
        tokenizer = tiktoken.get_encoding("gpt2")
        print(f"[Tokenizer] tiktoken gpt2, vocab={MODEL_CONFIG['vocab_size']}")
    else:
        MODEL_CONFIG["vocab_size"] = 256

    MODEL_CONFIG["context_length"] = args.ctx_len

    print("[Loading model...]")
    model = BigModel(MODEL_CONFIG).to(device=device, dtype=dtype)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"])
    print(f"[Model loaded] {model.num_params():,} params")
    print(
        f"  epoch={ck.get('epoch', '?')}  step={ck.get('step', '?')}  loss={ck.get('loss', float('nan')):.4f}"
    )

    problems = load_test_problems(
        "data/newfacade_LeetCodeDataset", max_problems=args.num_problems
    )
    print(f"[Loaded {len(problems)} test problems]")

    total_passed = 0
    total_tests = 0

    for problem in problems:
        task_id = problem.get("task_id", "unknown")
        difficulty = problem.get("difficulty", "?")
        query = problem.get("query", "")
        test_code = problem.get("test", "")

        print(f"\n{'=' * 60}")
        print(f"# {task_id} [{difficulty}]")
        print(f"{'=' * 60}")

        prompt = query + "\n" if tokenizer is None else query

        print(f"[Prompt (first 200 chars)]\n{prompt[:200]}...\n")

        output = generate(
            model,
            prompt,
            args.max_tokens,
            device,
            dtype,
            temperature=args.temperature,
            top_k=args.top_k,
            tokenizer=tokenizer,
        )

        print(f"[Model output (first 500 chars)]\n{output[:500]}...\n")

        code = extract_python_code(output)
        if code:
            print(f"[Extracted code (first 300 chars)]\n{code[:300]}...\n")
        else:
            print("[Extracted code] NONE\n")
            total_tests += 1
            print("Test 1: FAIL (no code)")
            continue

        passed, result = run_python_code(code, test_code)
        total_tests += 1
        if passed:
            total_passed += 1
            print("Test 1: PASS")
        else:
            print(f"Test 1: FAIL")
            print(f"  Output: {result[:300]}")

    print(f"\n{'=' * 60}")
    pass_rate = total_passed / max(total_tests, 1)
    print(f"OVERALL: {total_passed}/{total_tests} = {pass_rate:.2%}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
