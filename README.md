# Real-Time Tiny Model Trainer (Test-Time Training via SFT + RL)

## Context

The goal is a **router** that holds many specialized tiny models. When a problem arrives:
1. Router selects top-K most similar tiny models and tries them
2. Evaluates each using **prediction loss** on the problem's support examples
3. If any model solves it (low loss) → use its answer
4. If none can solve it (all losses too high) → generate synthetic curriculum data (easy→medium→hard) and train a **new tiny model in real-time** — including at test/inference time
5. New tiny model is added to the router's collection for future reuse

This behavior (routing + deciding to train) is learned during **SFT + RL training**, so the router automatically does it at test time too.

---

## Architecture

```
New Problem (with support examples)
         ↓
Router
  ├─ problem_encoder: raw_input → 64-dim embedding
  ├─ model_index: (model_id → specialty_embedding)
  │
  └─ Step 1: similarity_score(problem_emb, each model_emb) → rank all models
             ↓
  └─ Step 2: Try top-K=3 most similar tiny models
             For each: forward(problem.raw_input) → prediction
                       loss = MSE(prediction, problem.support_labels)
             ↓
  └─ Step 3: best_loss = min(losses)
             ├── best_loss < SOLVE_THRESHOLD → return best model's answer ✓
             └── best_loss >= SOLVE_THRESHOLD → "cannot solve" → train new model
                       ↓
             CurriculumGenerator → easy/medium/hard synthetic data
                       ↓
             CurriculumTrainer → 3-level warm-start training
             FeelingTracker   → monitors loss/accuracy per level
                       ↓
             New TinyModel added to Router's model bank
                       ↓
             New TinyModel.infer(problem) → Answer
```

---

## Problem Format

Each problem includes **support examples** (a few labeled input-output pairs) used to evaluate whether existing tiny models can solve it:

```python
@dataclass
class Problem:
    raw_input: np.ndarray          # the actual query input, padded to MAX_DIM=64
    support_X: np.ndarray          # shape (n_support, input_dim): example inputs
    support_y: np.ndarray          # shape (n_support, output_dim): expected outputs
    description: str = ""
    metadata: dict = field(default_factory=dict)
```

**Sequence prediction example:**
- `raw_input` = `[0., 1., 2., 3., 0., 0., 0., 0.]` (query: what comes next?)
- `support_X` = `[[0.,1.], [1.,2.], [2.,3.]]` (support: known examples of the pattern)
- `support_y` = `[[2.], [3.], [4.]]` (expected next elements)

The router tests each tiny model on `(support_X, support_y)` and computes MSE loss. Low loss = model understands this pattern = can solve.

---

## Project Structure

```
mulmodel/
├── README.md                        # this plan
├── requirements.txt
├── main.py                          # demo: solve 3 problems, show routing + training
├── train_policy.py                  # SFT + RL training entry point
│
├── core/
│   ├── problem.py                   # Problem dataclass (raw_input, support_X, support_y)
│   ├── answer.py                    # Answer (value, confidence, source, was_trained)
│   └── registry.py                  # model_id ↔ specialty_embedding registry
│
├── router/
│   ├── encoder.py                   # ProblemEncoder: raw_input → 64-dim embedding
│   ├── similarity.py                # similarity_score(problem_emb, model_emb)
│   ├── model_index.py               # ModelIndex: stores (model_id → specialty_emb)
│   └── router.py                    # Router: top-K selection + loss-based evaluation
│
├── tiny_model/
│   └── model.py                     # TinyModel: small MLP (1K-100K params)
│
├── bank/
│   └── model_bank.py                # ModelBank: store/retrieve TinyModels + embeddings
│
├── curriculum/
│   ├── feeling.py                   # FeelingTracker: loss/accuracy gates level advance
│   ├── generator.py                 # Abstract CurriculumGenerator base
│   └── trainer.py                   # CurriculumTrainer: 3-level warm-start training
│
├── data_generators/
│   ├── base.py                      # Abstract generator interface
│   ├── sequence_prediction.py       # Demo domain: [0]→[0,1]→[0,1,2,3,4]
│   ├── arithmetic.py                # Arithmetic operations
│   ├── pattern_matching.py          # Binary pattern rules
│   └── selector.py                  # GeneratorSelector: picks right generator for problem
│
├── policy/
│   ├── sft_trainer.py               # SFT: train router on demonstrations
│   └── rl_trainer.py                # RL: REINFORCE on routing + training decisions
│
├── system/
│   └── pipeline.py                  # RTTrainerPipeline: orchestrates everything
│
└── tests/
    ├── test_router.py
    ├── test_curriculum.py
    ├── test_feeling.py
    ├── test_tiny_model.py
    ├── test_model_bank.py
    └── test_pipeline.py
```

---

## Key Modules

### `router/router.py` — The Core Router

```python
class Router:
    """
    Holds reference to ModelBank and ModelIndex.
    Implements: select top-K by similarity, evaluate by loss, decide train/reuse.
    """
    TOP_K = 3
    SOLVE_THRESHOLD = 0.05   # MSE loss below this = "solved"

    def try_existing(self, problem: Problem) -> Tuple[Optional[TinyModel], float]:
        """
        1. Encode problem → embedding
        2. Score all models in bank by cosine similarity to problem embedding
        3. Try top-K: compute MSE(model(support_X), support_y) for each
        4. Return (best_model, best_loss)
        """

    def needs_new_model(self, best_loss: float) -> bool:
        return best_loss >= self.SOLVE_THRESHOLD
```

### `router/encoder.py` — Problem Encoder (Rule-Based)

No LLM needed. Uses hand-crafted statistical features:

```python
class ProblemEncoder:
    """
    Encodes a Problem into a 64-dim float vector:
    [0:16]  raw values (first 16 elements of raw_input)
    [16:32] statistical features (mean, std, min, max, diffs stats)
    [32:48] structural features (length, monotonicity, periodicity)
    [48:64] support statistics (support_X mean, std, label mean, std)

    All normalized to [-1, 1]. No learned parameters.
    """
    def encode(self, problem: Problem) -> np.ndarray: ...  # shape (64,)
```

The **specialty embedding** for each TinyModel is computed from the training data it saw:
```python
def compute_specialty_embedding(support_X: np.ndarray, support_y: np.ndarray) -> np.ndarray:
    # Same encoding applied to model's training data → 64-dim embedding
    # Stored in ModelIndex when model is registered
```

This means: problems that look structurally similar to a model's training data will score high similarity — without needing an LLM.

### `curriculum/feeling.py` — FeelingTracker

```python
class FeelingTracker:
    """
    Tracks learning progress. Gates advancement to next curriculum level.

    is_ready_to_advance(level) → True when:
      - val_accuracy >= {0: 0.85, 1: 0.80, 2: 0.75}[level]
      - OR stagnation: improvement < 0.5%/epoch AND min epochs passed
      - OR MAX_EPOCHS {0:20, 1:30, 2:50}[level] reached

    get_feeling_vector() → np.ndarray shape (9,)
      [easy_loss, easy_acc, med_loss, med_acc, hard_loss, hard_acc,
       epochs_easy, epochs_med, epochs_hard]
    """
```

### `curriculum/trainer.py` — CurriculumTrainer (Runs at Test Time)

```python
class CurriculumTrainer:
    """
    Trains a TinyModel through easy → medium → hard levels.
    Weights warm-start between levels (not reset). LR halved each level.

    Typical: 3-15 seconds on CPU for a Micro TinyModel (~10K params).

    for level in [0, 1, 2]:
        X, y = generator.generate(level, n_train=500)
        X_val, y_val = generator.generate(level, n_val=100)
        for epoch in range(MAX_EPOCHS[level]):
            train_epoch(model, optimizer, X, y)
            val_loss, val_acc = evaluate(model, X_val, y_val)
            feeling.record(level, epoch, train_loss, val_loss, val_acc)
            if feeling.is_ready_to_advance(level): break
        lr *= 0.5    # fine-tune at next level
    return model, feeling
    """
```

### `data_generators/sequence_prediction.py` — Demo Domain

```
Level 0 (Easy):   X=[x₀ padded to 8], y=[x₁]       single element → next
Level 1 (Medium): X=[x₀,x₁ padded to 8], y=[x₂]    short seq → next
Level 2 (Hard):   X=[x₀,...,x₆ padded], y=[x₇]     full seq → next

Patterns (randomly sampled per example):
  linear+1: [0,1,2,3,...], linear+2: [0,2,4,6,...],
  geometric×2: [1,2,4,8,...], squares: [0,1,4,9,...], fibonacci

Values normalized to [-1,1]. Pure Python/NumPy, no LLM.
```

### `system/pipeline.py` — RTTrainerPipeline

```python
class RTTrainerPipeline:
    def solve(self, problem: Problem) -> Answer:
        """
        Same code path at training time AND test time.

        1. router.try_existing(problem) → (best_model, best_loss)
        2a. best_loss < SOLVE_THRESHOLD:
              answer = best_model.infer(problem.raw_input)
              return Answer(source=f"bank:{model_id}", was_trained=False)
        2b. best_loss >= SOLVE_THRESHOLD:
              generator = GeneratorSelector.select(problem)
              model, feeling = CurriculumTrainer(generator).train(problem)
              new_id = bank.register(model, specialty_emb, feeling)
              answer = model.infer(problem.raw_input)
              return Answer(source=f"newly_trained:{new_id}", was_trained=True)
        """
```

---

## SFT + RL Training

### What Gets Trained
- `Router.encoder` weights (if made learnable) — or stays rule-based for v1
- `Router.similarity` function — could be a small dot-product MLP
- The routing policy: when to route vs. trigger training

### SFT Phase
Demonstrations of correct behavior generated synthetically:
```
Demo 1: problem_type_A exists in bank → correct action = route to existing model
Demo 2: new problem_type_B, no model in bank → correct action = trigger training
```
Train with cross-entropy loss on correct actions. No human labeling needed.

### RL Phase (REINFORCE)
```
State:   [problem_embedding (64) | bank_presence_vec (64)] = 128 dims
Action:  "route to model_id" OR "train new"
Reward:
  +1.0  correct routing + answer correct (loss < threshold)
  +0.5  trained new model + answer correct
  -0.5  routed but answer wrong (model couldn't solve it)
  -1.0  trained new model + answer still wrong
  -0.02/sec latency penalty for training time

Update: REINFORCE every 32 episodes. Baseline = EMA of recent rewards.
```

### Why This Teaches Test-Time Training
The RL reward is based on **answer correctness**. To get +1.0 at test time, the model must either:
- Successfully route to an existing model, OR
- Train a new model that gets the right answer

The policy learns this naturally — there's no explicit "test-time training" switch. It just does what gets rewarded.

---

## The Feeling Mechanism

```
Level 0 (Easy):
  Epoch 1: val_acc=0.22 → "can't feel the pattern"
  Epoch 5: val_acc=0.87 → "got easy patterns ✓" → advance

Level 1 (Medium):  [warm-starts from Level 0]
  Epoch 8: val_acc=0.81 → "stronger feeling ✓" → advance

Level 2 (Hard):  [warm-starts from Level 1]
  Epoch 10: val_acc=0.76 → "final model ready ✓"

feeling_vector = [0.18, 0.87, 0.21, 0.81, 0.35, 0.76, 5, 8, 10]
```

Feeling vector stored as metadata in the bank alongside each TinyModel.

---

## Implementation Order

```
Phase 1 — Core types:
  core/problem.py, core/answer.py, core/registry.py

Phase 2 — Data generators (start with sequence_prediction, it's the demo):
  data_generators/base.py, sequence_prediction.py, arithmetic.py,
  pattern_matching.py, selector.py

Phase 3 — TinyModel:
  tiny_model/model.py

Phase 4 — Curriculum (the test-time training engine):
  curriculum/feeling.py, curriculum/generator.py, curriculum/trainer.py

Phase 5 — Router:
  router/encoder.py, router/similarity.py,
  router/model_index.py, router/router.py

Phase 6 — Bank:
  bank/model_bank.py

Phase 7 — Policy training:
  policy/sft_trainer.py, policy/rl_trainer.py

Phase 8 — Orchestration + demo:
  system/pipeline.py, train_policy.py, main.py

Phase 9 — Tests:
  tests/ (all files)
```

---

## Verification

### Key Tests
```python
# 1. New problem → trains tiny model in solve()
p = Problem(raw_input=[0,1,2,3,...], support_X=..., support_y=...)
a = pipeline.solve(p)
assert a.was_trained == True

# 2. Same domain → reuses existing model (no training)
p2 = Problem(raw_input=[2,4,6,8,...], support_X=..., support_y=...)
a2 = pipeline.solve(p2)
assert a2.was_trained == False   # bank hit

# 3. Loss threshold check
loss = router.evaluate(existing_model, p)
assert loss < 0.05  # solved on second call

# 4. Curriculum levels all run
assert len(feeling.records) == 3  # easy, medium, hard
```

### Demo (`main.py` output)
```
Problem 1 (new sequence): trains tiny model... answer=4.1  [was_trained=True,  ~8s]
Problem 2 (same domain):  routed to bank...   answer=10.0 [was_trained=False, ~1ms]
Problem 3 (new binary):   trains tiny model... answer=1.0  [was_trained=True,  ~6s]
```

---

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pytest>=7.0.0
```

No external LLM APIs. All synthetic data is rule-based (pure Python + NumPy).
