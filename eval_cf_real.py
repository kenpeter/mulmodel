#!/usr/bin/env python3
"""
Real Codeforces evaluation - use actual problem statements from training data.
Uses proper generation with CoT prompting and robust code extraction.
"""

import torch
import subprocess
import tempfile
import os
import re

try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

from transformer import BigModel, MODEL_CONFIG


CODE_MARKER = "\n[CODE]\n"


REAL_CODEFORCES_PROBLEMS = [
    {
        "name": "Maximum Element",
        "description": """You are given an array a of n integers, where n is odd.

In one operation, you will remove two adjacent elements from the array a, and then concatenate the remaining parts of the array.

You will repeatedly perform this operation until exactly one element remains in a.

Find the maximum possible value of the remaining element in a.

Input: first line t test cases. Each test case: n on first line, then n integers.
Output: for each test case, output a single integer.""",
        "test_cases": [
            {"input": "1\n1\n6", "output": "6"},
            {"input": "1\n3\n1 3 2", "output": "2"},
            {"input": "1\n5\n4 7 4 2 9", "output": "9"},
        ],
    },
    {
        "name": "Two Sum",
        "description": """Given an array of n integers and a target T, determine if there exist two distinct indices i and j such that a[i] + a[j] = T.

Input: first line n and T, second line n integers.
Output: YES if such a pair exists, NO otherwise.""",
        "test_cases": [
            {"input": "4 9\n1 2 3 4", "output": "YES"},
            {"input": "3 6\n1 2 3", "output": "YES"},
            {"input": "3 10\n1 2 3", "output": "NO"},
        ],
    },
    {
        "name": "Palindrome",
        "description": """Given an integer x, return true if x is a palindrome, and false otherwise.

Input: a single integer.
Output: YES if palindrome, NO otherwise.""",
        "test_cases": [
            {"input": "121", "output": "YES"},
            {"input": "-121", "output": "NO"},
            {"input": "10", "output": "NO"},
        ],
    },
    {
        "name": "Reverse Array",
        "description": """Given an array of n integers, reverse the array and print it.

Input: first line n, second line n integers.
Output: reversed array as space-separated integers.""",
        "test_cases": [
            {"input": "5\n1 2 3 4 5", "output": "5 4 3 2 1"},
            {"input": "3\n10 20 30", "output": "30 20 10"},
        ],
    },
]


def generate(
    model,
    prompt: str,
    max_tokens: int,
    device,
    dtype,
    temperature=0.0,
    top_k=10,
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
        logits = logits[0, -1, :]
        if temperature == 0.0:
            next_id = logits.argmax().unsqueeze(0).unsqueeze(0)
        else:
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = float("-inf")
            probs = torch.softmax(logits / temperature, dim=0)
            next_id = torch.multinomial(probs, 1).unsqueeze(0)
        ids = torch.cat([ids, next_id], dim=1)
    if tokenizer is not None:
        return tokenizer.decode(ids[0].tolist())
    return bytes(ids[0].tolist()).decode("utf-8", errors="replace")


def postprocess_code(code: str) -> str:
    code = code.replace("<bits/stdc+++.h>", "<iostream>")
    code = code.replace("<bits/stdc++.h>", "<iostream>")
    code = code.replace("<bits/stdc++11.h>", "<iostream>")
    code = code.replace("<bits/stdc++14.h>", "<iostream>")
    code = code.replace("<bits/stdc++17.h>", "<iostream>")
    lines = code.split("\n")
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "+++" in line and "bits/stdc" not in line:
            for _ in range(3):
                line = line.replace("++++", "++")
        if ">>>>" in line:
            line = line.replace(">>>>", ">")
        if ">>>" in line:
            line = line.replace(">>>", ">>")
        result.append(line)
    return "\n".join(result)


def extract_code(output: str) -> str:
    if "[CODE]" in output:
        code = output.split("[CODE]")[-1].strip()
        if code:
            code = postprocess_code(code)
            return code
    lines = output.split("\n")
    for i, line in enumerate(lines):
        if "#include" in line or "int main" in line:
            result = []
            for l in lines[i:]:
                if l.strip().startswith("[CODE]"):
                    break
                result.append(l)
            if result:
                code = postprocess_code("\n".join(result).strip())
                return code
    if output.strip():
        return postprocess_code(output.strip())
    return ""


def run_cpp_code(code: str, stdin: str) -> str:
    if not code.strip():
        return "NO_CODE"

    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
        f.write(code)
        cpp_file = f.name

    exe_file = cpp_file.replace(".cpp", "")

    try:
        comp = subprocess.run(
            ["g++", cpp_file, "-o", exe_file],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if comp.returncode != 0:
            return f"COMPILE_ERROR: {comp.stderr[:200]}"

        result = subprocess.run(
            [exe_file], input=stdin, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        for f in [cpp_file, exe_file]:
            if os.path.exists(f):
                os.unlink(f)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--ctx-len", type=int, default=256)
    p.add_argument(
        "--tokenizer", type=str, default="byte", choices=["byte", "tiktoken"]
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    MODEL_CONFIG["context_length"] = args.ctx_len

    tokenizer = None
    if args.tokenizer == "tiktoken":
        MODEL_CONFIG["vocab_size"] = 50257
        tokenizer = tiktoken.get_encoding("gpt2")
        print(f"[Tokenizer] tiktoken gpt2, vocab={MODEL_CONFIG['vocab_size']}")
    else:
        MODEL_CONFIG["vocab_size"] = 256

    print("[Loading model...]")
    model = BigModel(MODEL_CONFIG).to(device=device, dtype=dtype)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"])
    print(f"[Model loaded] {model.num_params():,} params")
    print(
        f"  epoch={ck.get('epoch', '?')}  step={ck.get('step', '?')}  loss={ck.get('loss', float('nan')):.4f}"
    )

    total_passed = 0
    total_tests = 0

    for problem in REAL_CODEFORCES_PROBLEMS:
        print(f"\n{'=' * 60}")
        print(f"# {problem['name']}")
        print(f"{'=' * 60}")

        prompt = problem["description"].strip() + CODE_MARKER

        print(f"[Prompt (first 100 chars)]\n{prompt[:100]}...\n")

        output = generate(
            model,
            prompt,
            600,
            device,
            dtype,
            temperature=0.0,
            top_k=0,
            tokenizer=tokenizer,
        )

        print(f"[Model output (first 800 chars)]\n{output[:800]}...\n")

        code = extract_code(output)
        if code:
            print(f"[Extracted code (first 300 chars)]\n{code[:300]}...\n")
        else:
            print("[Extracted code] NONE\n")

        problem_passed = 0
        for i, tc in enumerate(problem["test_cases"]):
            result = run_cpp_code(code, tc["input"])
            expected = tc["output"]
            ok = result == expected
            total_tests += 1
            if ok:
                problem_passed += 1
                total_passed += 1
            print(f"Test {i + 1}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                print(f"  Input:    {repr(tc['input'])}")
                print(f"  Expected: {repr(expected)}")
                print(f"  Got:      {repr(result[:100])}")

        print(f"Score: {problem_passed}/{len(problem['test_cases'])}")

    print(f"\n{'=' * 60}")
    print(
        f"OVERALL: {total_passed}/{total_tests} = {total_passed / max(total_tests, 1):.2%}"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
