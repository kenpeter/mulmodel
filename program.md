# mulmodel autoresearch

**GOAL: Solve real LeetCode problems. pass_rate ≥ 50%.**

**RULE: Keep it simple, clean, small.** If a change adds complexity without improving pass_rate, don't do it. Smaller model, simpler code, fewer tricks — always prefer the lightweight approach.

This is an experiment to have the LLM do its own research on the mulmodel project.

## Dataset: newfacade/LeetCodeDataset

Training uses the [LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset) (arXiv:2504.14655):
- 2,869 Python LeetCode problems with rich metadata (difficulty, tags, release dates)
- 100+ test cases per problem
- Temporal split: pre-July 2024 = train, post-July 2024 = test
- Data format: JSONL with `query`, `response`, `completion`, `test`, `starter_code`, `problem_description`

Located at: `data/newfacade_LeetCodeDataset/leetcode_train.jsonl`

## How to Run

### Code Agent (opencode)

If you're running this in a code agent (like opencode), **do NOT use start.sh**. Run directly:

```
opencode run "read program.md auto research"
```

Or in continuous mode:
```
opencode run "read program.md auto research"
```
(Repeat as needed)

### Phone / Openclaw

If running from phone via openclaw, use `start.sh`:
```
./start.sh
```

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Check for existing checkpoint**: If `checkpoints/latest.pt` exists, we'll resume from it. If not, start fresh.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `train.py` — training loop, optimizer, hyperparameters, data loading.
   - `transformer.py` — model architecture (`MODEL_CONFIG`, `BigModel`, attention, blocks).
4. **Verify data exists**: Check that `data/newfacade_LeetCodeDataset/` contains `leetcode_train.jsonl`. If empty, tell the human.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU for a **fixed time budget of 5 minutes** (wall clock training time). Launch it as:

```
# Resume from checkpoint if exists, otherwise start fresh
python train.py --time-limit 300 $([ -f checkpoints/latest.pt ] && echo "--resume") > run.log 2>&1
```

Or explicitly:
```
python train.py --time-limit 300 --resume > run.log 2>&1
```

**What you CAN do:**
- Modify `train.py` — training loop, optimizer, hyperparameters, batch size, learning rate, scheduler, etc.
- Modify `transformer.py` — model architecture, `MODEL_CONFIG`, attention, layers, activations, etc.

**What you CANNOT do:**
- Touch anything in `data/` — training data is read-only.
- Touch `analysis.ipynb`, `results.tsv`, `requirements.txt`, `pyproject.toml`, or any agent/config files.
- Modify the `--time-limit` flag logic — the 5-minute budget is sacred.
- Install new packages.

**THE ONLY METRIC THAT MATTERS: Can the model solve real LeetCode problems?**

- **Don't care about loss or perplexity** - these don't predict code solving ability
- **Goal: pass_rate ≥ 50%** on real LeetCode problems
- **Judge: Python execution + test cases** - run code against inputs, check outputs
- Use real LeetCode problems from the test set (post-July 2024)

**VRAM** is a soft constraint — the GPU has 12 GB. OOM = crash = discard.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**The first run**: Always establish the baseline first — run the script as-is before making any changes.

## Output format

The script prints this at the end of every timed run:

```
---
train_loss:       0.843200
perplexity:       2.324
training_seconds: 300.1
peak_vram_mb:     8590.2
```

Extract the key metric:

```
grep "^perplexity:\|^peak_vram_mb:" run.log
```

## True eval — LeetCode functional correctness

After each training run, evaluate whether the model can solve real LeetCode problems:

```
python eval_leetcode.py --checkpoint checkpoints/latest.pt --ctx-len 512 --tokenizer tiktoken
```

For byte-level models:
```
python eval_leetcode.py --checkpoint checkpoints/latest.pt --ctx-len 512 --tokenizer byte
```

This will:
1. Load problems from `data/newfacade_LeetCodeDataset/leetcode_test.jsonl`
2. Generate Python solutions using the model
3. Extract code from output (handles ```python blocks, [CODE] marker)
4. Run the solution against the test harness
5. Calculate pass_rate = passed_tests / total_tests

**The primary metric is pass_rate ≥ 50%.** Perplexity is secondary.

**Judge: Python execution + test case verification.** If the code runs and produces correct output, it passes. No LLM judge.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	perplexity	pass_rate	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. perplexity achieved (e.g. 2.324) — use 0.000 for crashes
3. pass_rate as percentage (e.g. 80.0 for 4/5 tests passed) — use 0.00 for crashes
4. peak memory in GB, round to .1f (e.g. 8.6 — divide peak_vram_mb by 1024) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	perplexity	pass_rate	memory_gb	status	description
a1b2c3d	4.782	0.00	8.4	keep	baseline
b2c3d4e	3.890	25.00	8.6	keep	increase LR to 1e-4
c3d4e5f	4.920	0.00	8.4	discard	switch to GeLU activation
d4e5f6g	0.000	0.00	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar14`). Training runs in **5-minute cycles**: train → stop → review → improve → resume. Repeat forever.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Run training for 5 minutes (it saves a checkpoint automatically):
   ```
   python train.py --time-limit 300 --resume > run.log 2>&1
   ```
3. Read out the results: `grep "^perplexity:\|^peak_vram_mb:" run.log`
4. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
5. **Run eval with REAL LeetCode problems**:
   ```
   python eval_leetcode.py --checkpoint checkpoints/latest.pt --ctx-len 512 --tokenizer tiktoken
   ```
   This runs generated Python against test cases. Reports pass_rate.
6. Record the results in `results.tsv` (do NOT commit this file)
7. **Review** — focus on pass_rate improvement:
   - If pass_rate improved → keep the commit
   - If pass_rate dropped → revert and try different approach
   - Focus on architectural changes that help code generation
7. Implement the chosen improvement in `train.py` or `transformer.py`.
8. git commit the change.
9. **Resume** training from the last checkpoint — do NOT start from scratch:
   ```
   python train.py --time-limit 300 --resume > run.log 2>&1
   ```
10. If train_loss improved (lower) after the new cycle, keep the commit.
11. If train_loss is equal or worse, `git reset --hard HEAD~1` to revert the code change, then resume from the checkpoint again with a different idea.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. If you run out of ideas, think harder — try different architectures, optimizers, learning rate schedules, batch sizes. The loop runs until the human interrupts you, period.

**Timeout**: If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM or bug), use judgment: fix trivial issues and re-run. If the idea is fundamentally broken, log "crash" and move on.
