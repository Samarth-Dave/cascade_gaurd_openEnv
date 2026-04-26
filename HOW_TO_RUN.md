# How to Run CascadeGuard

## 1. Install

```bash
cd metaV1/cascade_guard
pip install -e .
pip install openenv-core pydantic fastapi uvicorn
```

For GRPO training (optional):
```bash
pip install trl transformers accelerate peft bitsandbytes
# For memory-efficient training on T4 (optional):
pip install unsloth
```

---

## 2. Start the Environment Server

```bash
python -m uvicorn cascade_guard.server.app:app --host 0.0.0.0 --port 8000 --reload
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

---

## 3. Run Heuristic Baseline (Quick Sanity Check)

```bash
python inference.py --task task_easy --policy heuristic --seeds 3
```

Expected output:
```
[task_easy | seed=42] score=0.55  actions=[harden, recover, recover]
[task_easy | seed=1042] score=0.56  actions=[recover, shed_load, recover]
[task_easy | seed=2042] score=0.53  actions=[harden, recover, wait]
Mean score: 0.547
```

---

## 4. Run Demo Inference (Judge-Facing Mode)

```bash
python inference.py --task task_medium --policy groq --demo-mode --seeds 1
```

This renders a live step-by-step display with:
- Color-coded node health bars
- Narrative phase banners (PRE_STORM → STORM_ONSET → CRISIS_PEAK → RECOVERY)
- LLM reasoning excerpts  
- Budget gauge and radius coverage events
- Final score breakdown

For the new Metro City scenario:
```bash
python inference.py --task task_real_city --policy heuristic --demo-mode
```

---

## 5. Run Multi-Seed Evaluation

```bash
python scripts/eval_multiseed.py --tasks task_easy task_medium task_real_city --seeds 3
```

This produces `logs/eval_results.csv` and prints a summary table.

---

## 6. Run GRPO Training

### Minimum viable settings (should produce score > 0.50):
```bash
python training/train_grpo.py \
  --task task_easy \
  --steps 200 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
```

### In notebook mode (see `test vs trained.ipynb`):
Set in the model-loading cell:
```python
USE_HEAVY_MODEL = True
RAW_ADAPTER_REPO = "https://huggingface.co/samarthdave0305/cascadeguard-trained/tree/main"
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
```
Then run all cells. The adapter tensors are loaded on top of the DeepSeek base model.

**Expected outcome**: `grpo` policy score ≥ 0.55 on task_easy (vs 0.50 heuristic baseline).

---

## Score Targets

| Score | Meaning |
|---|---|
| < 0.50 | Worse than doing nothing |
| 0.50 | Neutral baseline |
| 0.55–0.65 | Basic awareness |
| 0.65–0.75 | Competent — usually right |
| 0.75–0.85 | Strong — outperforms heuristic |
| > 0.85 | Expert-level management |
