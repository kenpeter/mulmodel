import torch
import re
from transformer import BigModel, MODEL_CONFIG

MODEL_CONFIG["context_length"] = 256
device = torch.device("cuda")
dtype = torch.bfloat16
model = BigModel(MODEL_CONFIG).to(device=device, dtype=dtype)
ck = torch.load("checkpoints/latest.pt", map_location=device)
model.load_state_dict(ck["model"])


@torch.no_grad()
def generate(prompt, max_tokens=300, temp=0.3):
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


def extract_code(output):
    import re

    lines = output.split("\n")
    code_lines = []
    in_code = False

    for line in lines:
        if "```python" in line or "```py" in line or "```" in line:
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines)

    code_indicators = [
        r"^import ",
        r"^from ",
        r"^def ",
        r"^class ",
        r"^if ",
        r"^for ",
        r"^while ",
        r"^print\(",
        r"^return ",
        r"^    if ",
        r"^    for ",
        r"^    while ",
        r"^    return ",
        r"^    print\(",
        r"^n =",
        r"^arr =",
        r"^a =",
        r"^b =",
        r"^result",
        r"^target",
        r"^nums",
    ]
    pattern = re.compile("|".join(code_indicators))

    code_lines = []
    for line in lines:
        if pattern.match(line.strip()):
            code_lines.append(line)
        elif code_lines and line.strip() and not line.strip().startswith("#"):
            if any(c in line for c in "()[]=+-*/<>") or line.startswith(" " * 4):
                code_lines.append(line)

    if code_lines:
        cleaned = []
        for line in code_lines:
            if "===" in line or "---" in line:
                continue
            if "Okay" in line or "Let" in line or "Thus" in line or "So" in line:
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    return output


# Test FizzBuzz problem with simpler prompt
prompt = """n = int(input())
"""
print("=== Prompt ===")
print(repr(prompt))
print()

output = generate(prompt)
print("=== Raw Output ===")
print(output[:500])
print()

code = extract_code(output)
print("=== Extracted Code ===")
print(repr(code[:200]))
