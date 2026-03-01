# mulmodel — Competition Math LoRA Test-Time Trainer

A system that trains **LoRA specialist patches on-demand** when it sees a new type of competition math problem.  BigModel (frozen 101M-param transformer) provides general reasoning; LoRA patches (~800K params each) specialise it for each problem template at test time.

---

## How It Works

```
New competition math problem arrives
           ↓
LoRAPipeline.solve(problem_text)
           ↓
    Infer template type from keywords
           ↓
    ┌──── template already in bank? ────┐
    │                                   │
   YES                                  NO
    │                                   │
  ROUTE (instant)                    SPAWN
  attach existing LoRA patch    generate 60 similar problems
  BigModel + LoRA → predict     train new LoRA patch (100 steps)
           ↓                    save patch to bank
         Answer                         ↓
                                BigModel + LoRA → predict
                                       ↓
                                     Answer
```

**SPAWN** happens once per template type, then that patch is reused forever (ROUTE).

---

## Architecture

```
BigModel (frozen, 101M params)
  256-dim hidden  ·  128 layers  ·  8 heads  ·  BERT backbone
  Checkpoint: big_model_data/big_model.pt

LoRA Patch (per template type, ~800K params)
  Applied to last 32 attention layers (Q, K, V projections)
  + AnswerHead: Linear(256→64)→ReLU→Linear(64→1)
  Saved to: bank_data/lora_N/lora.pt

Competition Math Templates (5 built-in)
  ModularArithmetic  —  x^k ≡ c (mod p)
  LinearEquation     —  ax + b = c
  ArithmeticSeries   —  sum of first n terms, diff d
  GeometricSeries    —  sum of first n terms, ratio r
  Combinatorics      —  C(n,k) mod m
```

---

## Training Progress

### Step 1 — Install dependencies  ✅

```bash
pip install -r requirements.txt
# torch, numpy, pytest, transformers, datasets
```

### Step 2 — Pretrain BigModel  ⬜ (do this first)

No data download needed — Wikipedia is streamed on the fly, math sequences are synthetic.

```bash
# GPU (recommended — 4070 12GB, ~2 min/epoch)
python -m big_model.pretrain --epochs 20 --steps 200

# CPU smoke-test only (very slow)
python -m big_model.pretrain --epochs 2 --steps 10
```

What happens each epoch:
- **Text**: Wikipedia articles byte-encoded → 15% tokens masked → predict masked bytes (cross-entropy)
- **Math**: synthetic linear/geometric/quadratic sequences → 15% values masked → predict (MSE)
- Checkpoint saved to `big_model_data/big_model.pt` after every epoch

Progress output:
```
Epoch   1/20  math_loss=0.3412  text_loss=4.8901
Epoch   2/20  math_loss=0.2103  text_loss=4.2310
...
Epoch  20/20  math_loss=0.0341  text_loss=3.1042
[BigModel] Done. Checkpoint: big_model_data/big_model.pt
```

### Step 3 — Run the competition math demo  ⬜

```bash
python main.py
```

Expected output (GPU):
```
Device: cuda   train_steps=100   n_similar=60
════════════════════════════════════════════════════════════════
#1 Modular arithmetic  → SPAWN  patch=lora_0  loss=...  t=~5s
#2 Linear equation     → SPAWN  patch=lora_1  loss=...  t=~5s
#3 Geometric series    → SPAWN  patch=lora_2  loss=...  t=~5s
#4 Modular again       → ROUTE  sim=1.000               t=<0.5s
#5 Linear again        → ROUTE  sim=1.000               t=<0.5s

LoRA Bank: 3 patches   Problems solved: 5
  lora_0  template=modular_arithmetic  solves=1
  lora_1  template=linear_equation     solves=1
  lora_2  template=geometric_series    solves=0
```

On second run: all 5 problems ROUTE instantly (bank loaded from disk).

---

## Project Structure

```
mulmodel/
│
├── main.py                          # competition math demo entry point
├── download_data.py                 # optional: download Wikitext-103 + competition_math
├── requirements.txt
│
├── big_model/
│   ├── transformer.py               # BigModel: 256-wide 128-layer BERT (101M params)
│   ├── lora.py                      # LoRAAdapter, LoRAPatch, AnswerHead
│   ├── pretrain.py                  # pretraining: masked token modeling
│   └── encoder.py                   # BigModelEncoder: encode() → 256-dim embedding
│
├── curriculum/
│   ├── lora_trainer.py              # LoRATrainer: freeze BigModel, train LoRA patch
│   ├── trainer.py                   # CurriculumTrainer (TinyModel, legacy)
│   └── feeling.py                   # FeelingTracker (TinyModel, legacy)
│
├── data_generators/
│   ├── competition_math.py          # 5 templates + generate_similar() + infer_template()
│   └── ...                          # legacy sequence/text generators
│
├── system/
│   ├── lora_pipeline.py             # LoRAPipeline: SPAWN / ROUTE orchestration
│   └── pipeline.py                  # RTTrainerPipeline (TinyModel, legacy)
│
├── bank/
│   ├── model_bank.py                # ModelBank (TinyModel, legacy)
│   └── persistence.py               # BankPersistence (TinyModel, legacy)
│
├── router/
│   ├── router.py                    # Router (TinyModel, legacy)
│   ├── encoder.py                   # ProblemEncoder: 64-dim rule-based (legacy)
│   └── model_index.py               # ModelIndex: cosine similarity index
│
└── tests/
    └── ...                          # pytest suite
```

---

## File Relationships (current active path)

```
main.py
  └── system/lora_pipeline.py        ← orchestrates everything
        ├── big_model/transformer.py  ← BigModel backbone (frozen)
        ├── big_model/lora.py         ← LoRAPatch (hooks into BigModel)
        ├── curriculum/lora_trainer.py← trains LoRA patch on support examples
        └── data_generators/
              competition_math.py    ← generates training problems per template
```

---

## Bank Layout (disk)

```
bank_data/
  lora_index.json          ← { counter, patches: {lora_0: {template, solve_count}, ...} }
  lora_0/
    lora.pt                ← LoRAPatch state dict (~3MB)
    embedding.npy          ← 256-dim specialty embedding (for future embedding routing)
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
  2. Infer template type from keywords (congruent/mod → Modular, solve → Linear, ...)
  3. If matching template in bank → ROUTE (attach patch, re-encode with LoRA, predict)
  4. Else → SPAWN (generate_similar → LoRATrainer.train → register → predict)
```

After BigModel pretraining, embedding similarity (step 1) will naturally separate template types, enabling routing for truly novel problems with no keyword match.

---

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pytest>=7.0.0
transformers>=4.30.0    # BertConfig, BertModel
datasets>=2.14.0        # Wikipedia streaming for pretraining
```
