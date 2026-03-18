# Autoresearch Progress - mar18

## What We've Done

### Setup
- Created branch `autoresearch/mar18` from main
- Using Codeforces CoT data (data/code/codeforces_cots)
- 202M param model (emb_dim=1024, n_layers=16, n_heads=16, ctx=512)
- Trained with: lr=5e-4, batch_size=4, ctx_len=256, weight_decay=0.01

### Training Progress
- Started: perplexity ~3.05
- Current: perplexity ~2.05 (plateau after ~50 cycles)
- Each cycle: 5 minutes training
- VRAM: ~4.3 GB

### Experiments Tried
1. Baseline (lr=3e-4) → perplexity 3.05
2. Increase LR to 5e-4 → slight improvement
3. Add 100 warmup steps → minor improvement  
4. Reduce weight decay 0.1→0.01 → significant improvement
5. Increased n_layers to 24 → OOM/failed
6. Various architectural changes reverted due to issues

### Current Model Behavior
- Generates chain-of-thought explanations well
- Struggles to generate actual code when given problem description
- Training data format: problem description → CoT explanation → C++ code
- Model learns to generate explanations, not directly code

## The Problem

**Perplexity doesn't predict code-solving ability!**
- Model can have low perplexity but fail at generating working code
- Need to evaluate on **actual Codeforces problem solving**

## Next Move

### Immediate
1. Run eval with real Codeforces problems to get pass_rate
2. Try different prompt formats:
   - Provide code template instead of problem description
   - Use few-shot examples from training data
3. Consider:
   - Greedy decoding (temp=0) for more deterministic output
   - Training specifically on code generation (not CoT)
   - Larger model capacity if VRAM allows

### Long-term
1. Implement opencode as LLM judge for better evaluation
2. Try different architectures that favor code generation
3. Consider training on code-only data vs CoT data

## Files
- `program.md` - experiment protocol
- `eval.py` - evaluation script  
- `results.tsv` - experiment results
- `checkpoints/latest.pt` - current checkpoint
