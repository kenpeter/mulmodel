# mulmodel autoresearch

This is an experiment to have the LLM do its own research on the mulmodel project.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `big_model/pretrain.py` — the file you modify. Model architecture, optimizer, hyperparameters, training loop. Everything is fair game.
4. **Verify data exists**: Check that `data/` contains training data. If empty, tell the human.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU for a **fixed time budget of 5 minutes** (wall clock training time). Launch it as:

```
python -m big_model.pretrain --time-limit 300 > run.log 2>&1
```

**What you CAN do:**
- Modify `big_model/pretrain.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, batch size, learning rate, etc.

**What you CANNOT do:**
- Modify the data pipeline or training data in `data/`.
- Modify the `--time-limit` flag logic — the 5-minute budget is sacred.
- Install new packages beyond what's in `big_model/requirements.txt`.

**The goal: get `train_loss` as low as possible.** Since time is fixed at 5 min per run, focus on better architectures, optimizers, and hyperparams that make the most of the budget.

**VRAM** is a soft constraint — the GPU has 12 GB. OOM = crash = discard.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**The first run**: Always establish the baseline first — run the script as-is before making any changes.

## Output format

The script prints this at the end of every timed run:

```
---
train_loss:       0.843200
training_seconds: 300.1
peak_vram_mb:     8590.2
```

Extract the key metric:

```
grep "^train_loss:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	train_loss	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. train_loss achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 8.6 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	train_loss	memory_gb	status	description
a1b2c3d	1.560000	8.4	keep	baseline
b2c3d4e	1.480000	8.6	keep	increase LR to 1e-4
c3d4e5f	1.590000	8.4	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar14`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `big_model/pretrain.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python -m big_model.pretrain --time-limit 300 > run.log 2>&1`
5. Read out the results: `grep "^train_loss:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't fix it after a few attempts, give up and discard.
7. Record the results in `results.tsv` (do NOT commit this file — leave it untracked)
8. If train_loss improved (lower), advance the branch keeping the git commit
9. If train_loss is equal or worse, `git reset --hard HEAD~1` to revert

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. If you run out of ideas, think harder — try different architectures, optimizers, learning rate schedules, batch sizes. The loop runs until the human interrupts you, period.

**Timeout**: If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM or bug), use judgment: fix trivial issues and re-run. If the idea is fundamentally broken, log "crash" and move on.
