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

### Step 3 — Pretrain BigModel  🔄 In Progress

**Architecture:** GPT-style causal language model (next-token prediction on byte sequences)

```bash
# GPU (recommended — 4070 12GB)
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

#### Training Progress

| Epoch | code_loss | Notes |
|-------|-----------|-------|
| 1 | 5.52 | start (≈ log(256) = random) |
| 20 | 1.63 | first 20-epoch run |
| 943 | 0.935 | interrupted at epoch 943/1000 |

Loss is steadily decreasing — longer training = better generations.

#### Resuming pretraining

Training resumes **automatically** — just re-run:

```bash
python -m big_model.pretrain --epochs 20 --steps 200
```

Loads `big_model_latest.pt` + `train_state.json` and continues from the last completed epoch. Use `--fresh` to start from scratch.

#### Chat with the model

```bash
python -m big_model.chat
python -m big_model.chat --temp 0.7 --top_k 40 --tokens 512
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
