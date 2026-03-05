# mulmodel — Coding Problem LoRA Test-Time Trainer

A system that trains **LoRA specialist patches on-demand** when it sees a new type of coding problem. BigModel (frozen 101M-param transformer) provides general reasoning; LoRA patches (~800K params each) specialise it for each problem type at test time.

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
BigModel (frozen, 101M params)
  256-dim hidden  ·  128 layers  ·  8 heads  ·  BERT backbone
  Checkpoints: big_model_data/big_model_latest.pt  /  big_model_data/big_model_best.pt

LoRA Patch (per problem type, ~800K params)
  Applied to last 32 attention layers (Q, K, V projections)
  + AnswerHead: Linear(256→64)→ReLU→Linear(64→1)
  Saved to: bank_data/lora_N/lora.pt
```

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
python download_data.py --apps        # APPS only
python download_data.py --codeforces  # Codeforces-CoTs only
python download_data.py --leetcode    # LeetCode (requires HF access request)
```

| Dataset | Problems | Notes |
|---------|----------|-------|
| `deepmind/code_contests` | 11k problems + 13M solutions | Codeforces/AtCoder, used for AlphaCode |
| `codeparrot/apps` | 10k problems | Intro → competition level, with test cases |
| `open-r1/codeforces-cots` | ~100k samples | Codeforces + chain-of-thought reasoning |
| `Nan-Do/leetcode_contests` | 4.7M submissions | LeetCode style (gated — request access first) |

Re-running resumes automatically — HuggingFace caches downloaded shards.

### Step 3 — Pretrain BigModel  ⬜

```bash
# GPU (recommended — 4070 12GB, ~2 min/epoch)
python -m big_model.pretrain --epochs 20 --steps 200

# CPU smoke-test only (very slow)
python -m big_model.pretrain --epochs 2 --steps 10
```

What happens each epoch:
- **Code datasets**: problems + solutions byte-encoded → 15% tokens masked → predict masked bytes (cross-entropy)
- Two checkpoints saved after every epoch:
  - `big_model_data/big_model_latest.pt` — always updated
  - `big_model_data/big_model_best.pt` — updated only when total loss improves

Progress output:
```
Epoch   1/20  code_loss=4.8901  math_loss=4.7210  [best]
Epoch   2/20  code_loss=4.2310  math_loss=4.1005  [best]
...
Epoch  20/20  code_loss=3.1042  math_loss=3.0891
[BigModel] Done. Latest: big_model_data/big_model_latest.pt  Best: big_model_data/big_model_best.pt
```

#### Resuming pretraining

Training resumes **automatically** — no extra flag needed. Just re-run the same command:

```bash
python -m big_model.pretrain --epochs 20 --steps 200
```

On startup the script checks for `big_model_data/big_model_latest.pt` and `big_model_data/train_state.json`. If found, it loads the weights and picks up from where it left off (correct epoch, step count, and best loss). If not found, it starts from scratch.

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
│   ├── transformer.py               # BigModel: 256-wide 128-layer BERT (101M params)
│   ├── lora.py                      # LoRAAdapter, LoRAPatch, AnswerHead
│   ├── pretrain.py                  # pretraining: masked token modeling on code datasets
│   └── encoder.py                   # BigModelEncoder: encode() → 256-dim embedding
│
├── curriculum/
│   ├── lora_trainer.py              # LoRATrainer: freeze BigModel, train LoRA patch
│   └── ...                          # legacy trainers
│
├── data_generators/
│   ├── competition_math.py          # math problem templates (legacy fallback)
│   └── ...                          # other generators
│
├── system/
│   ├── lora_pipeline.py             # LoRAPipeline: SPAWN / ROUTE orchestration
│   └── ...                          # legacy pipeline
│
└── tests/
    └── ...                          # pytest suite
```

---

## File Relationships (active path)

```
main.py
  └── system/lora_pipeline.py        ← orchestrates everything
        ├── big_model/transformer.py  ← BigModel backbone (frozen)
        ├── big_model/lora.py         ← LoRAPatch (hooks into BigModel)
        ├── curriculum/lora_trainer.py← trains LoRA patch on support examples
        └── data_generators/          ← generates training problems per type
```

---

## Bank Layout (disk)

```
bank_data/
  lora_index.json          ← { counter, patches: {lora_0: {template, solve_count}, ...} }
  lora_0/
    lora.pt                ← LoRAPatch state dict (~3MB)
    embedding.npy          ← 256-dim specialty embedding
  lora_1/
    ...
```

Autosaved every 5 problems. Reloaded on next run automatically.

---

## LoRA Training Details

| Parameter | GPU (cuda) | CPU |
|-----------|-----------|-----|
| `n_train_steps` | 100 | 20 |
| `n_similar` | 60 | 10 |
| Time per SPAWN | ~5s | ~50s |
| Time per ROUTE | <0.5s | <0.5s |

LoRA only modifies last 32 of 128 layers (Q, K, V). BigModel weights never changed.

---

## Routing Logic

```
solve(problem_text):
  1. Encode text → 256-dim embedding (base BigModel, no LoRA)
  2. Infer problem type from keywords
  3. If matching type in bank → ROUTE (attach patch, re-encode with LoRA, predict)
  4. Else → SPAWN (generate_similar → LoRATrainer.train → register → predict)
```

---

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pytest>=7.0.0
transformers>=4.30.0    # BertConfig, BertModel
datasets>=2.14.0        # HuggingFace datasets for pretraining
```
