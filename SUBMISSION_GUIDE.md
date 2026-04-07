# CascadeGuard Submission Guide

This guide assumes your submission root is the cascade_guard folder.

## 1) Required Files and Placement

1. Keep inference script at:
   - [inference.py](inference.py)
2. Keep OpenEnv config at:
   - [openenv.yaml](openenv.yaml)
3. Keep Dockerfile at:
   - [Dockerfile](Dockerfile)
4. Keep validator script at:
   - [scripts/validate-submission.sh](scripts/validate-submission.sh)

## 2) Mandatory Environment Variables

Set these before running inference:

1. API_BASE_URL
2. MODEL_NAME
3. HF_TOKEN
4. LOCAL_IMAGE_NAME (recommended for local docker image selection)

PowerShell example:

```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN = "hf_xxx"
$env:LOCAL_IMAGE_NAME = "cascade-guard:latest"
```

## 3) Compliance Expectations (What You Must Keep)

1. All LLM calls must use OpenAI client.
   - Already satisfied in [inference.py](inference.py) via AsyncOpenAI.
2. Structured stdout must only emit:
   - [START]
   - [STEP]
   - [END]
3. Inference runtime should be under 20 minutes.
4. Must run on low-resource machine profile (2 vCPU, 8GB RAM).

## 4) Local Pre-Submission Runbook

Run from cascade_guard folder.

1. OpenEnv compliance:

```powershell
openenv validate
```

2. Task and grader sanity (3+ tasks with valid grader outputs):

```powershell
python .\verify_tasks.py
```

3. Docker build check:

```powershell
docker build -t cascade-guard:latest .
```

4. Baseline inference check:

```powershell
python .\inference.py
```

5. Full validator (Linux/macOS shell, Git Bash, or WSL):

```bash
bash scripts/validate-submission.sh https://your-space.hf.space .
```

## 5) Hugging Face Space Upload Steps

1. Create a new Space on Hugging Face:
   - SDK: Docker
   - Visibility: your preference
2. Push this cascade_guard folder as the Space repo contents.
3. In Space Settings -> Variables and secrets, add:
   - API_BASE_URL
   - MODEL_NAME
   - HF_TOKEN
4. Make sure Space starts and endpoint responds:
   - POST https://<your-space>.hf.space/reset must return HTTP 200.

## 6) GitHub Repo Setup

1. Push submission code to your GitHub repo.
2. Ensure repo includes the same submission folder content.
3. Confirm links work publicly (or as allowed by hackathon rules).

## 7) Final Submission Form (What to Paste)

1. GitHub Repository URL:
   - https://github.com/<username>/<repo>
2. Hugging Face Space URL:
   - https://huggingface.co/spaces/<username>/<space>
3. Checklist confirmation text (recommended):

```text
Confirmed:
- inference.py is present in submission root.
- API_BASE_URL, MODEL_NAME, HF_TOKEN are configured.
- Defaults are only set for API_BASE_URL and MODEL_NAME.
- All LLM calls use OpenAI client.
- Structured stdout follows START/STEP/END format.
- openenv validate passes.
- Docker build passes.
- verify_tasks.py passes across multiple tasks/graders.
- Pre-submission validator script was run successfully.
```

## 8) Judging Phases (Quick Reminder)

1. Phase 1: Automated validation gate.
2. Phase 2: Agentic evaluation and variance checks.
3. Phase 3: Human review for utility/creativity/exploit resistance.

## 9) Disqualification Triggers to Avoid

1. Space not deployable or /reset not responding.
2. Missing baseline inference script.
3. Non-functional or constant-score graders.
4. Plagiarized/trivial environment modification.
