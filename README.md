# mulmodel — Coding Problem LoRA Test-Time Trainer

A system that trains **LoRA specialist patches on-demand** when it sees a new type of coding problem. BigModel (frozen 1.21B-param transformer) provides general reasoning; LoRA patches specialise it for each problem type at test time.

---

## How It Works

```
New coding problem arrives
           ↓
LoRAPipeline.solve(problem_text)
           ↓
    Infer problem type from keywords
           ↓
    ┌──── type already in bank? ────┐
    │                               │
   YES                              NO
    │                               │
  ROUTE (instant)                SPAWN
  attach existing LoRA patch   generate 60 similar problems
  BigModel + LoRA → predict    train new LoRA patch (100 steps)
           ↓                   save patch to bank
         Answer                        ↓
                               BigModel + LoRA → predict
                                       ↓
                                     Answer
```

**SPAWN** happens once per problem type, then that patch is reused forever (ROUTE).

---

## Architecture

```
BigModel (1.21B params)
  1024-dim hidden  ·  96 layers  ·  16 heads  ·  GPT-style causal LM
  Context window: 512 bytes
  Checkpoints: big_model_data/big_model_latest.pt  /  big_model_data/big_model_best.pt

LoRA Patch (per problem type)
  Applied to attention layers (Q, K, V projections)
  Saved to: bank_data/lora_N/lora.pt
```

---

## How 1.21B params fit in 12 GB VRAM

This is the key question. A naive FP32 training run of 1.21B params needs:

| Thing | Size |
|---|---|
| FP32 model weights | 4.8 GB |
| FP32 gradients | 4.8 GB |
| FP32 Adam m + v states | 9.6 GB |
| **Total** | **~19 GB** — does not fit |

We use three tricks to bring this under 12 GB:

### Trick 1 — BF16 model weights on GPU

Store the model in **BF16** (2 bytes per param) instead of FP32 (4 bytes):

```
1.21B params × 2 bytes = 2.4 GB  (instead of 4.8 GB)
```

BF16 has the same exponent range as FP32 (no gradient scaling needed), just lower mantissa precision. Good enough for training.

### Trick 2 — FP32 optimizer states on CPU RAM

Adam needs two momentum tensors per parameter (m and v). At FP32 that's:

```
1.21B params × 8 bytes = 9.7 GB
```

Instead of putting these on the GPU, **AdamW runs entirely on CPU RAM**. The training step looks like:

```
GPU (BF16)                       CPU RAM (FP32)
──────────────────────────────   ──────────────────────────────
forward pass (BF16)          →
loss.backward() → BF16 grads →   copy grads to FP32
                                 clip_grad_norm_()
                                 AdamW.step()  ← optimizer lives here
GPU params updated           ←   copy updated params back as BF16
```

This keeps ~9.7 GB of optimizer state off the GPU entirely. With 90 GB of system RAM, there's plenty of room.

### Trick 3 — Gradient checkpointing

Normally PyTorch saves every layer's intermediate activations for the backward pass. With 96 layers that's a lot. Gradient checkpointing discards them and **recomputes** each layer's activations during backward instead.

Trade-off: ~2× more compute per step, but activation memory drops from O(layers) to O(1).

### Result

| Thing | Location | Size |
|---|---|---|
| BF16 model weights | GPU VRAM | **2.4 GB** |
| BF16 gradients (peak, during backward) | GPU VRAM | **2.4 GB** |
| Activations (gradient checkpointing) | GPU VRAM | **~0.2 GB** |
| FP32 Adam m + v | CPU RAM | 9.7 GB |
| **Peak VRAM total** | | **~5 GB / 12 GB** |

---

## Training Progress

### Step 1 — Install dependencies  ✅

```bash
pip install -r requirements.txt
```

### Step 2 — Download coding datasets  ⬜

```bash
python download_data.py               # all datasets (~6.5GB total)
python download_data.py --contests    # DeepMind CodeContests only
python download_data.py --codeforces  # Codeforces-CoTs only
```

| Dataset | Problems | Notes |
|---------|----------|-------|
| `deepmind/code_contests` | 11k problems + 13M solutions | Codeforces/AtCoder |
| `open-r1/codeforces-cots` | ~100k samples | Codeforces + chain-of-thought |

### Step 3 — Pretrain BigModel  🔄 In Progress

```bash
# GPU (recommended — 4070 Ti 12GB)
python -m big_model.pretrain --epochs 20 --steps 200

# CPU smoke-test only (very slow)
python -m big_model.pretrain --epochs 2 --steps 10
```

`--epochs N` means **train N more epochs** from wherever training left off.

What happens each epoch:
- Coding problems + solutions byte-encoded → predict next byte (causal cross-entropy)
- Two checkpoints saved after every epoch:
  - `big_model_data/big_model_latest.pt` — always updated
  - `big_model_data/big_model_best.pt` — updated only when loss improves
- Checkpoints saved in **FP32** (from the CPU master copy) for full precision

#### Training Progress

| Epoch | code_loss | Notes |
|-------|-----------|-------|
| 1 | 5.52 | start (≈ log(256) = random) |
| 20 | 1.63 | first run, old 256-wide architecture |
| 943 | 0.935 | old architecture, 128 layers × 256 wide |
| — | — | **architecture upgrade: 96 layers × 1024 wide, 1.21B params** |
| 81 | 0.771 | first run on new architecture (resumed from compatible partial ckpt) |

#### Resuming pretraining

Training resumes **automatically** — just re-run:

```bash
python -m big_model.pretrain --epochs 20 --steps 200
```

Loads `big_model_latest.pt` + `train_state.json` and continues from the last completed epoch.
If the checkpoint is from a different architecture, it starts fresh automatically.

#### Chat / code completion

```bash
python -m big_model.chat
python -m big_model.chat --temp 0.7 --top_k 40 --tokens 200
```

Enter a raw code prefix (not a question) — the model completes it:

```
>>> def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        ...
```

### Step 4 — Run the demo  ⬜

```bash
python main.py
```

---

## Project Structure

```
mulmodel/
│
├── main.py                          # demo entry point
├── download_data.py                 # download coding datasets (resumable)
├── requirements.txt
│
├── big_model/
│   ├── transformer.py               # BigModel: 1024-wide 96-layer GPT (1.21B params)
│   ├── pretrain.py                  # pretraining: causal LM, BF16+CPU-optimizer
│   ├── chat.py                      # code completion REPL
│   ├── lora.py                      # LoRAAdapter, LoRAPatch, AnswerHead
│   └── encoder.py                   # BigModelEncoder: encode() → 1024-dim embedding
│
├── big_model_data/
│   ├── big_model_latest.pt          # checkpoint saved every epoch (FP32)
│   ├── big_model_best.pt            # checkpoint saved on loss improvement (FP32)
│   └── train_state.json             # epoch, step, best_loss for resuming
│
├── system/
│   └── lora_pipeline.py             # LoRAPipeline: SPAWN / ROUTE orchestration
│
└── tests/
    └── ...
```

---

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pytest>=7.0.0
transformers>=4.30.0    # GPT2Config, GPT2Model
datasets>=2.14.0        # HuggingFace datasets for pretraining
accelerate>=0.26.0      # installed but not used for training dispatch
```
