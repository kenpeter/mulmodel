import torch
import subprocess, tempfile, os
from transformer import BigModel, MODEL_CONFIG

MODEL_CONFIG["context_length"] = 256
device = torch.device("cuda")
dtype = torch.bfloat16
model = BigModel(MODEL_CONFIG).to(device=device, dtype=dtype)
ck = torch.load("checkpoints/latest.pt", map_location=device)
model.load_state_dict(ck["model"])


@torch.no_grad()
def generate(prompt, max_tokens=200, temp=0.3):
    model.eval()
    ids = torch.tensor(
        list(prompt.encode("utf-8")), dtype=torch.long, device=device
    ).unsqueeze(0)
    for _ in range(max_tokens):
        inp = ids[:, -256:]
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(inp)
        probs = torch.softmax(logits[0, -1, :] / temp, dim=0)
        next_id = torch.multinomial(probs, 1).unsqueeze(0)
        ids = torch.cat([ids, next_id], dim=1)
        if ids[0, -1].item() == 0:
            break
    return bytes(ids[0].tolist()).decode("utf-8", errors="replace")


def run_code(code, stdin):
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            ["python", fname], input=stdin, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        os.unlink(fname)


def extract_code(output):
    lines = output.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if "```python" in line or "```py" in line:
            in_code = True
            continue
        if "```" in line and in_code:
            break
        if in_code:
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines)
    for line in lines:
        if "def " in line or "print(" in line or "import " in line:
            code_lines.append(line)
    return "\n".join(code_lines) if code_lines else output


# Test Two Sum problem
prompt = """Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
Input: first line contains n, second line contains n integers, third line contains target.
Output: two indices (0-based) that sum to target, separated by space.
"""
prompt = prompt.strip() + "\n"

output = generate(prompt)
print("=== RAW OUTPUT ===")
print(output[:500])
print()

code = extract_code(output)
print("=== EXTRACTED CODE ===")
print(repr(code))
print()

result = run_code(code, "5\n2 7 11 15\n9")
print("=== RUN RESULT ===")
print(repr(result))
