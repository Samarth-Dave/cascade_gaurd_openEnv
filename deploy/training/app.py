"""
CascadeGuard — SFT + GRPO Training
Converted from notebook for Hugging Face Spaces (Gradio + GPU).
Runs training in background; Gradio UI shows live logs.

CHANGELOG vs original:
  • patch_train_grpo() — rewrites 'dtype' → 'torch_dtype' in train_grpo.py
    before the subprocess launches.  Fixes:
      TypeError: Qwen2ForCausalLM.__init__() got an unexpected keyword argument 'dtype'
    Root-cause: transformers ≥ 4.45 removed the bare `dtype` constructor kwarg
    for Qwen2-based models (DeepSeek-R1-Distill-Qwen-32B uses Qwen2ForCausalLM).
    The correct kwarg is `torch_dtype`.  The SFT step already used `torch_dtype=`
    correctly; the GRPO script's `mkw` dict used the old `'dtype':` key.
"""

import os, sys, subprocess, time, threading, math, json, re
import matplotlib
matplotlib.use('Agg')  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import gradio as gr

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = '/home/user/app'
REPO_URL     = 'https://github.com/Samarth-Dave/cascade_gaurd_openEnv'
REPO_DIR     = os.path.join(BASE_DIR, 'cascade_gaurd_openEnv')
OUTPUT_DIR   = os.path.join(REPO_DIR, 'training_outputs')
CURVES_DIR   = os.path.join(REPO_DIR, 'training_curves')
RESULTS_CSV  = os.path.join(REPO_DIR, 'cascadeguard_eval_results.csv')
CACHE_DIR    = '/tmp/cascadeguard_hf_cache'
OSM_CACHE    = '/tmp/osmnx_cache'
SFT_DATASET_PATH = os.path.join(REPO_DIR, 'cascadeguard_sft_dataset.csv')

# ── HF Token ──────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get('HF_TOKEN', '')

# ── Training config ───────────────────────────────────────────────────────────
MODEL        = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
SFT_STEPS    = 150
GRPO_STEPS   = 120
STATES       = 64
GENERATIONS  = 4
LORA_R       = 16
LR_SFT       = 1.5e-5
LR_GRPO      = 5e-6
MAX_SEQ_LEN  = 1024
MAX_COMP_LEN = 256
RUN_EVAL     = True

SFT_SAVE_PATH  = os.path.join(OUTPUT_DIR, 'sft', 'final')
GRPO_SAVE_PATH = os.path.join(OUTPUT_DIR, 'grpo', 'final')

# ── Environment flags ─────────────────────────────────────────────────────────
os.environ['TORCHINDUCTOR_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['HF_HOME']               = CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE']    = CACHE_DIR
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose         = False

for d in [CACHE_DIR, OSM_CACHE]:
    os.makedirs(d, exist_ok=True)

def ensure_output_dirs():
    for d in [OUTPUT_DIR, CURVES_DIR, SFT_SAVE_PATH]:
        os.makedirs(d, exist_ok=True)

# ── Shared state ──────────────────────────────────────────────────────────────
log_lines     = []
training_done = False
sft_records   = []
grpo_records  = []

def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)
    if len(log_lines) > 1000:
        log_lines.pop(0)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 — verify GPU
# ═══════════════════════════════════════════════════════════════════════════════
def verify_gpu():
    log("=" * 60)
    log("STEP 0 — Verify GPU")
    log("=" * 60)
    r = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    log(r.stdout[:600])
    log(f"PyTorch       : {torch.__version__}")
    log(f"PyTorch CUDA  : {torch.version.cuda}")

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log(f"GPU           : {p.name}")
        log(f"VRAM          : {p.total_memory / 1e9:.1f} GB")
        return

    if 'NVIDIA' in r.stdout:
        log("")
        log("⚠️ nvidia-smi sees a GPU, but torch.cuda.is_available() is False.")
        log(f"   PyTorch build: {torch.__version__} (CUDA {torch.version.cuda})")
        log(f"   Fix: pin a compatible torch in requirements.txt:")
        log(f"     --extra-index-url https://download.pytorch.org/whl/cu124")
        log(f"     torch==2.6.0+cu124")
        raise RuntimeError("CUDA driver / PyTorch build mismatch — see logs above.")
    raise RuntimeError("No GPU found. Attach GPU hardware in Space settings.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — verify dependencies
# ═══════════════════════════════════════════════════════════════════════════════
def install_deps():
    log("=" * 60)
    log("STEP 1 — Verifying dependencies")
    log("=" * 60)

    required = {
        'torch':        '>=2.5',
        'transformers': '>=4.46',
        'trl':          '>=0.12',
        'peft':         '>=0.14',
        'accelerate':   '>=1.0',
        'datasets':     '>=2.20',
        'bitsandbytes': '>=0.43',
        'osmnx':        '>=1.9',
        'pandas':       'any',
        'matplotlib':   'any',
    }
    import importlib
    missing = []
    for pkg in required:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, '__version__', '?')
            log(f"  {pkg:15s} {ver}")
        except ImportError:
            missing.append(pkg)
            log(f"  {pkg:15s} ❌ MISSING")

    if missing:
        log(f"⚠️ Missing packages: {missing}")
        raise ImportError(f"Missing required packages: {missing}")

    log("✅ All packages present.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — clone repo
# ═══════════════════════════════════════════════════════════════════════════════
def clone_repo():
    log("=" * 60)
    log("STEP 2 — Clone cascade_guard source from GitHub")
    log("=" * 60)
    from pathlib import Path

    repo = Path(REPO_DIR)
    if repo.is_dir() and (repo / '.git').exists():
        log(f"Repo present at {REPO_DIR} — running git pull ...")
        r = subprocess.run(
            ['git', '-C', REPO_DIR, 'pull', '--ff-only'],
            capture_output=True, text=True,
        )
        log(r.stdout.strip() or '(no output)')
        if r.returncode != 0:
            log(f'git pull stderr: {r.stderr.strip()}')
    else:
        log(f"Cloning {REPO_URL} → {REPO_DIR} ...")
        r = subprocess.run(
            ['git', 'clone', '--depth', '1', REPO_URL, REPO_DIR],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            log(f'git clone stderr: {r.stderr.strip()}')
            raise RuntimeError(f'git clone failed: {r.stderr.strip()}')
        log('Clone complete ✓')

    required_files = [
        'server/__init__.py',
        'server/cascade_environment.py',
        'server/graders.py',
        'models.py',
        'tasks.py',
        'reward.py',
        'geo_utils.py',
        'adversarial_attacker.py',
        'cascadeguard_sft_dataset.csv',
        'training/__init__.py',
        'training/train_grpo.py',
        'training/cot_prompt.py',
    ]
    missing = [f for f in required_files if not (repo / f).exists()]
    if missing:
        log("⚠️ Files MISSING from the GitHub remote:")
        for f in missing:
            log(f"     - {f}")
        raise FileNotFoundError(f"Missing files on GitHub remote: {missing}")
    log(f"✅ Repo ready ({len(required_files)} required files verified).")


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH — fix dtype → torch_dtype in train_grpo.py
# ═══════════════════════════════════════════════════════════════════════════════
def patch_train_grpo(train_script: str) -> None:
    """
    Surgically fix the `dtype` → `torch_dtype` bug in train_grpo.py.

    Root cause
    ----------
    In transformers ≥ 4.45, Qwen2ForCausalLM (which DeepSeek-R1-Distill-Qwen-32B
    uses) no longer accepts a bare `dtype` kwarg in its `__init__`.
    `from_pretrained` must receive `torch_dtype=` instead.

    The `load_model()` function in train_grpo.py builds a `mkw` dict that
    historically used `'dtype': torch.bfloat16`.  This patch rewrites:

        'dtype'  :  →  'torch_dtype'  :       (dict-key form)
        "dtype"  :  →  'torch_dtype'  :       (double-quote variant)
        dtype=   →  torch_dtype=              (keyword-argument form,
                                               guarded to only fire inside
                                               from_pretrained call sites)

    The patch is idempotent — running it twice leaves the file unchanged.
    """
    log("-" * 50)
    log("PATCH — fixing dtype → torch_dtype in train_grpo.py")

    with open(train_script, 'r', encoding='utf-8') as fh:
        src = fh.read()

    original = src
    
    # ── 1. Dict-key form: 'dtype': xxx  or  "dtype": xxx ──────────────────
    #    Matches only the bare key 'dtype' (not 'bnb_4bit_compute_dtype' etc.)
    src = re.sub(r"['\"]dtype['\"]\s*:", "'torch_dtype':", src)

    # ── 2. Keyword-argument form inside from_pretrained / model __init__ ──
    #    e.g.  AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)
    #    We only replace `dtype=` that is NOT preceded by an identifier char
    #    (to avoid corrupting names like `compute_dtype=` or `quant_dtype=`).
    src = re.sub(
        r'(?<![A-Za-z0-9_])dtype=(?!None)',
        'torch_dtype=',
        src,
    )

    # ── 3. mkw.update / mkw[...] assignments ──────────────────────────────
    #    e.g.  mkw.update({'dtype': ...})  or  mkw['dtype'] = ...
    #    Pattern 1 already handles the dict-key case.
    #    Handle  mkw['dtype']  just in case:
    src = re.sub(
        r"""\[(['"])dtype\1\]""",
        "['torch_dtype']",
        src,
    )

    # Fix older bundled GRPO scripts whose load_model() signature requires
    # max_seq_length, but whose main() call never forwards args.max_seq_len.
    def add_max_seq_length(match):
        block = match.group(0)
        if 'max_seq_length' in block:
            return block
        cache_match = re.search(r'\n(?P<indent>\s*)cache_dir\s*=', block)
        if cache_match:
            indent = cache_match.group('indent')
            return block[:cache_match.start()] + f"\n{indent}max_seq_length=args.max_seq_len," + block[cache_match.start():]
        close_match = re.search(r'\n(?P<indent>\s*)\)\s*$', block)
        if close_match:
            indent = close_match.group('indent') + '    '
            return block[:close_match.start()] + f"\n{indent}max_seq_length=args.max_seq_len," + block[close_match.start():]
        return block

    src = re.sub(
        r'model,\s*tokenizer,\s*peft_config,\s*backend,\s*torch_dtype\s*=\s*load_model\([\s\S]*?\n\s*\)',
        add_max_seq_length,
        src,
        count=1,
    )

    # ── FIX TRL 0.17 SFTTrainer API changes (SAFE PATCH) ──

# remove dataset_text_field ONLY inside SFTTrainer(...)
    src = re.sub(
        r'SFTTrainer\s*\((.*?)dataset_text_field\s*=\s*[^,]+,?',
        r'SFTTrainer(\1',
        src,
        flags=re.DOTALL
    )

# remove packing ONLY inside SFTTrainer(...)
    src = re.sub(
        r'SFTTrainer\s*\((.*?)packing\s*=\s*[^,]+,?',
        r'SFTTrainer(\1',
        src,
        flags=re.DOTALL
    )
    
    # remove max_seq_length ONLY inside SFTTrainer(...)
    src = re.sub(
        r'SFTTrainer\s*\((.*?)max_seq_length\s*=\s*[^,]+,?',
        r'SFTTrainer(\1',
        src,
        flags=re.DOTALL
    )
    
    # replace tokenizer= ONLY inside SFTTrainer(...)
    src = re.sub(
        r'SFTTrainer\s*\((.*?)tokenizer\s*=',
        r'SFTTrainer(\1processing_class=',
        src,
        flags=re.DOTALL
    )

    if src == original:
        log("  ℹ️  No train_grpo.py compatibility patch needed (already correct).")
    else:
        with open(train_script, 'w', encoding='utf-8') as fh:
            fh.write(src)
        # Count replacements for the log
        n = sum(1 for a, b in zip(original.splitlines(), src.splitlines()) if a != b)
        log(f"  ✅ Patched {n} line(s) in train_grpo.py compatibility fixes.")

    log("-" * 50)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — install package
# ═══════════════════════════════════════════════════════════════════════════════
def install_package():
    log("=" * 60)
    log("STEP 3 — Make cascade_guard package importable (symlink)")
    log("=" * 60)
    import importlib, shutil
    from pathlib import Path

    alias = Path(BASE_DIR) / 'cascade_guard'
    repo  = Path(REPO_DIR)

    if alias.is_symlink():
        alias.unlink()
    elif alias.exists():
        shutil.rmtree(alias)

    try:
        alias.symlink_to(repo, target_is_directory=True)
        log(f"Created symlink: {alias} → {repo}")
    except OSError as e:
        log(f"Symlink failed ({e}); copying directory instead ...")
        shutil.copytree(repo, alias, symlinks=False)
        log(f"Copied {repo} → {alias}")

    init = alias / '__init__.py'
    if not init.exists():
        init.write_text('"""cascade_guard package (auto-shim)."""\n')

    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    importlib.invalidate_caches()

    for mod in [m for m in list(sys.modules)
                if m == 'cascade_guard' or m.startswith('cascade_guard.')]:
        del sys.modules[mod]

    from cascade_guard.server.cascade_environment import CascadeEnvironment  # noqa
    from cascade_guard.models import CascadeAction                           # noqa
    from cascade_guard.tasks import TASK_CONFIGS                             # noqa
    log('✅ cascade_guard package ready (imports verified).')

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — pre-warm OSM data
# ═══════════════════════════════════════════════════════════════════════════════
def prewarm_osm():
    log("=" * 60)
    log("STEP 4 — Pre-warm OSM data")
    log("=" * 60)
    try:
        import osmnx as ox
        ox.settings.use_cache = True
        ox.settings.cache_folder = OSM_CACHE
        log(f"osmnx {ox.__version__} — cache: {OSM_CACHE}")
    except ImportError:
        log("osmnx not found — environment will use its own cache.")

    try:
        from cascade_guard.models import CascadeAction
        from cascade_guard.server.cascade_environment import CascadeEnvironment
        from cascade_guard.tasks import TASK_SEED_SPLITS

        tasks = list(TASK_SEED_SPLITS.keys())
        log(f"Tasks: {tasks}")
        t0 = time.perf_counter()
        for task_id in tasks[:3]:
            env  = CascadeEnvironment()
            seed = TASK_SEED_SPLITS[task_id]['train'][0]
            obs  = env.reset(task_id=task_id, seed=seed,
                             reward_mode='grpo', training_mode=True)
            env.step(CascadeAction(action_type='wait', target_node_id=None))
            log(f"  {task_id:30s} nodes={len(obs.nodes)} ✓")
            del env
        log(f"✅ OSM pre-warm complete in {time.perf_counter()-t0:.1f}s")
    except Exception as e:
        log(f"⚠️ Pre-warm raised: {e} — training will still proceed.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5/6/7 — load dataset + model
# ═══════════════════════════════════════════════════════════════════════════════
def load_dataset_and_model():
    log("=" * 60)
    log("STEP 5-7 — Load SFT dataset + model (4-bit + LoRA)")
    log("=" * 60)
    import pandas as pd
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    import bitsandbytes as bnb
    from bitsandbytes.nn import Params4bit, Int8Params

    assert os.path.exists(SFT_DATASET_PATH), \
        f"SFT CSV not found at {SFT_DATASET_PATH}"
    sft_raw = pd.read_csv(SFT_DATASET_PATH)
    sft_raw = sft_raw.dropna(subset=['system', 'prompt', 'completion'])
    log(f"SFT dataset: {len(sft_raw)} rows | tasks: {sft_raw['task'].value_counts().to_dict()}")

    log(f"Loading tokenizer for {MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True, cache_dir=CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    log("Tokenizer loaded ✓")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    log(f"Loading {MODEL} in 4-bit (bf16 compute) ...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,          # ← correct kwarg (NOT `dtype=`)
    )
    log(f"Base model loaded in {time.perf_counter()-t0:.0f}s ✓")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_R * 2,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05, bias='none',
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if isinstance(param, (Params4bit, Int8Params)):
            continue
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)

    model.print_trainable_parameters()
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"VRAM used: {used:.1f} / {total:.1f} GB")

    log("✅ Model ready.")
    return tokenizer, model, sft_raw

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — SFT training
# ═══════════════════════════════════════════════════════════════════════════════
def run_sft(tokenizer, model, sft_raw):
    log("=" * 60)
    log("STEP 8 — SFT Training (TRL SFTTrainer)")
    log("=" * 60)
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback

    def format_row(row):
        messages = [
            {'role': 'system',    'content': str(row['system']).strip()},
            {'role': 'user',      'content': str(row['prompt']).strip()},
            {'role': 'assistant', 'content': str(row['completion']).strip()},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = (
                f"<|system|>\n{messages[0]['content']}\n"
                f"<|user|>\n{messages[1]['content']}\n"
                f"<|assistant|>\n{messages[2]['content']}"
            )
        return {'text': text}

    sft_dataset = Dataset.from_list([format_row(row) for _, row in sft_raw.iterrows()])
    log(f"SFT dataset: {len(sft_dataset)} formatted examples")

    sft_args = SFTConfig(
        output_dir                  = f'{OUTPUT_DIR}/sft',
        max_steps                   = SFT_STEPS,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        learning_rate               = LR_SFT,
        warmup_ratio                = 0.05,
        lr_scheduler_type           = 'cosine',
        max_grad_norm               = 1.0,
        logging_steps               = 1,
        save_steps                  = max(SFT_STEPS, 1),
        report_to                   = 'none',
        bf16                        = True,
        fp16                        = False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    sft_records = []

    class SFTLogger(TrainerCallback):
        def on_log(self, _args, state, _control, logs=None, **kwargs):
            if logs:
                rec = {'step': state.global_step}
                for k, out in [('loss', 'loss'), ('train/loss', 'loss'), ('learning_rate', 'lr')]:
                    if k in logs and out not in rec:
                        v = logs[k]
                        rec[out] = float(v) if v is not None else float('nan')
                if len(rec) > 1:
                    sft_records.append(rec)
                    log(f"  SFT step {state.global_step} | loss={logs.get('loss', '?')}")

    for _n, _p in model.named_parameters():
        if _p.requires_grad and _p.dtype != torch.float32:
            _p.data = _p.data.to(torch.float32)

    trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=sft_dataset,
    processing_class=tokenizer,   # ✅ NEW (replaces tokenizer=)
    callbacks=[SFTLogger()],
    )
    trainer.train()
    log("✅ SFT training complete.")

    globals()['sft_records'].extend(sft_records)

    log(f"Saving SFT checkpoint to {SFT_SAVE_PATH} ...")
    trainer.save_model(SFT_SAVE_PATH)
    tokenizer.save_pretrained(SFT_SAVE_PATH)
    log("SFT checkpoint saved ✓")

    return sft_records

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9-10 — GRPO training (subprocess)
# ═══════════════════════════════════════════════════════════════════════════════
def fix_corrupted_train_grpo(train_script: str) -> None:
    """
    The train_grpo.py on the GitHub remote is Base64-encoded (corrupted).
    This function detects that and decodes it back to valid Python source.
    
    Detection: a valid Python file starts with '#', '\"\"\"', 'import', etc.
    A Base64 blob starts with a long run of alphanumeric chars with no spaces.
    """
    import base64

    log("-" * 50)
    log("CHECK — verifying train_grpo.py is valid Python (not Base64)")

    with open(train_script, 'rb') as fh:
        raw = fh.read()

    # Heuristic: if first 20 bytes have no spaces/newlines → likely Base64
    first_chunk = raw[:80].decode('ascii', errors='replace')
    is_base64 = (
        '\n' not in first_chunk[:40]
        and ' ' not in first_chunk[:40]
        and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' 
                for c in first_chunk[:40] if c != '\n')
    )

    if not is_base64:
        log("  ✅ train_grpo.py looks like valid Python — no decode needed.")
        log("-" * 50)
        return

    log("  ⚠️  Detected Base64-encoded train_grpo.py — decoding ...")
    try:
        # Strip any whitespace before decoding
        decoded = base64.b64decode(raw.strip())
        # Sanity check: decoded content should look like Python
        decoded_text = decoded.decode('utf-8', errors='replace')
        if 'import' not in decoded_text[:500] and 'def ' not in decoded_text[:500]:
            log("  ⚠️  Decoded content doesn't look like Python — aborting decode.")
            log("-" * 50)
            return
        with open(train_script, 'wb') as fh:
            fh.write(decoded)
        log(f"  ✅ Decoded successfully — {len(decoded)} bytes written.")
    except Exception as e:
        log(f"  ❌ Base64 decode failed: {e}")

    log("-" * 50)
# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9-10 — GRPO training (subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

def install_correct_train_grpo(train_script: str) -> None:
    """
    The train_grpo.py cloned from GitHub may be outdated or corrupted.
    We ship the correct version alongside app.py in the Space repo and
    overwrite the cloned copy before launching the subprocess.
    """
    import shutil

    log("-" * 50)
    log("INSTALL — overwriting train_grpo.py with Space-bundled version")

    # The correct file sits next to app.py in the Space
    space_copy = os.path.join(BASE_DIR, 'train_grpo.py')

    if not os.path.exists(space_copy):
        log(f"  ⚠️  No bundled train_grpo.py found at {space_copy} — using repo version.")
        log("-" * 50)
        return

    try:
        os.makedirs(os.path.dirname(train_script), exist_ok=True)
        shutil.copy2(space_copy, train_script)
        log(f"  ✅ Copied {space_copy} → {train_script}")
    except Exception as e:
        log(f"  ❌ Copy failed: {e} — using repo version.")

    log("-" * 50)
    
def run_grpo():
    log("=" * 60)
    log("STEP 9-10 — GRPO Training (train_grpo.py subprocess)")
    log("=" * 60)

    train_script = os.path.join(REPO_DIR, 'training', 'train_grpo.py')

    # ── STEP A: ensure the training/ directory exists ──────────────────────
    os.makedirs(os.path.dirname(train_script), exist_ok=True)

    # ── STEP B: overwrite broken GitHub copy with Space-bundled version ────
    try:
        install_correct_train_grpo(train_script)
    except Exception as e:
        log(f"⚠️ install_correct_train_grpo() raised: {e}")

    # ── STEP C: verify the file is actually there now ──────────────────────
    if not os.path.exists(train_script):
        log(f"⚠️ train_grpo.py not found at {train_script} — skipping GRPO.")
        return []

    # ── STEP D: patch dtype → torch_dtype (safety, idempotent) ────────────
    try:
        patch_train_grpo(train_script)
    except Exception as e:
        log(f"⚠️ patch_train_grpo() raised: {e} — proceeding anyway.")

    # ── STEP E: build and launch subprocess ────────────────────────────────
    grpo_output_dir = os.path.join(OUTPUT_DIR, 'grpo')

    cmd = [
        sys.executable, train_script,
        '--model',              MODEL,
        '--output-dir',         grpo_output_dir,
        '--cache-dir',          CACHE_DIR,
        '--steps',              str(GRPO_STEPS),
        '--states',             str(STATES),
        '--generations',        str(GENERATIONS),
        '--lora-r',             str(LORA_R),
        '--lr',                 str(LR_GRPO),
        '--max-seq-len',        str(MAX_SEQ_LEN),
        '--max-completion-len', str(MAX_COMP_LEN),
        '--repo-root',          REPO_DIR,
        '--no-unsloth',
    ]
    if not RUN_EVAL:
        cmd.append('--no-eval')
    if HF_TOKEN:
        cmd += ['--hf-token', HF_TOKEN]

    log(f"Running: {' '.join(cmd)}")

    grpo_records = []
    try:
        sub_env = {
            **os.environ,
            'PYTHONUNBUFFERED': '1',
            'PYTHONPATH': BASE_DIR + os.pathsep + os.environ.get('PYTHONPATH', ''),
        }
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=sub_env,
        )
        for line in proc.stdout:
            line = line.rstrip()
            log(line)
            m = re.search(r'step[_\s:=]+(\d+).*?reward[_\s:=]+([-\d.]+)', line, re.I)
            if m:
                try:
                    rec = {
                        'step':   SFT_STEPS + int(m.group(1)),
                        'reward': float(m.group(2)),
                    }
                    grpo_records.append(rec)
                    globals()['grpo_records'].append(rec)
                except ValueError:
                    pass
        proc.wait()
        if proc.returncode != 0:
            log(f"⚠️ GRPO subprocess exited with code {proc.returncode}")
        else:
            log("✅ GRPO training complete.")
    except Exception as e:
        import traceback
        log(f"⚠️ GRPO step error: {e}")
        log(traceback.format_exc())
        log("Skipping GRPO — SFT model will still be pushed.")

    return grpo_records

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 11 — plot curves
# ═══════════════════════════════════════════════════════════════════════════════
def plot_curves(sft_records, grpo_records):
    log("=" * 60)
    log("STEP 11 — Plotting training curves")
    log("=" * 60)
    all_records = sorted(sft_records + grpo_records, key=lambda r: r['step'])

    def extract(key, records=None):
        records = records or all_records
        xs = [r['step'] for r in records if key in r and not math.isnan(r.get(key, float('nan')))]
        ys = [r[key]    for r in records if key in r and not math.isnan(r.get(key, float('nan')))]
        return np.array(xs), np.array(ys)

    def smooth(ys, w=None):
        w = w or max(3, len(ys) // 10)
        return np.convolve(ys, np.ones(w) / w, mode='valid'), w

    # Loss curve
    xs_l, ys_l = extract('loss')
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if len(xs_l) > 0:
        sft_m  = xs_l <= SFT_STEPS
        grpo_m = xs_l  > SFT_STEPS
        if sft_m.any():
            ax.plot(xs_l[sft_m],  ys_l[sft_m],  color='#3a7ebf', lw=1, alpha=0.55, label='SFT loss')
        if grpo_m.any():
            ax.plot(xs_l[grpo_m], ys_l[grpo_m], color='#e05c5c', lw=1, alpha=0.55, label='GRPO loss')
        ys_sm, w = smooth(ys_l)
        ax.plot(xs_l[w-1:], ys_sm, color='#1a1a1a', lw=2, label=f'Smoothed (w={w})')
        ax.axvline(SFT_STEPS, color='#888', ls=':', lw=1.2, label='SFT→GRPO')
        ax.annotate(f'Final: {ys_l[-1]:.4f}', xy=(xs_l[-1], ys_l[-1]),
                    xytext=(-60, 12), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='#333'), fontsize=9)
    ax.set_title('CascadeGuard — Training Loss (SFT + GRPO)', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, ls='--')
    plt.tight_layout()
    for p in [f'{CURVES_DIR}/training_loss_curve.png', f'{OUTPUT_DIR}/training_loss_curve.png']:
        fig.savefig(p, dpi=160, bbox_inches='tight')
    plt.close()
    log("Saved training_loss_curve.png")

    # Reward curve
    xs_r, ys_r = extract('reward', grpo_records)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if len(xs_r) > 0:
        ax.plot(xs_r, ys_r, color='#3a7ebf', lw=1, alpha=0.55, label='Mean reward')
        ys_sm, w = smooth(ys_r)
        ax.plot(xs_r[w-1:], ys_sm, color='#003070', lw=2.5, label=f'Smoothed (w={w})')
        ax.axhline(0, color='gray', lw=0.8, ls=':', label='Baseline=0')
        ax.annotate(f'Final: {ys_r[-1]:.4f}', xy=(xs_r[-1], ys_r[-1]),
                    xytext=(-60, 12), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='#003070'), fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No reward data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
    ax.set_title('CascadeGuard GRPO — Training Reward', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Mean reward')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, ls='--')
    plt.tight_layout()
    for p in [f'{CURVES_DIR}/training_reward_curve.png', f'{OUTPUT_DIR}/training_reward_curve.png']:
        fig.savefig(p, dpi=160, bbox_inches='tight')
    plt.close()
    log("Saved training_reward_curve.png")

    with open(f'{CURVES_DIR}/training_log.json', 'w') as f:
        json.dump(all_records, f, indent=2, default=str)
    log("✅ Curves saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 13 — push model to HF Hub
# ═══════════════════════════════════════════════════════════════════════════════
def push_to_hub():
    log("=" * 60)
    log("STEP 13 — Push model to Hugging Face Hub")
    log("=" * 60)
    if not HF_TOKEN:
        log("⚠️ HF_TOKEN secret not set — skipping Hub push.")
        return

    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer

    login(token=HF_TOKEN)

    grpo_final = os.path.join(OUTPUT_DIR, 'grpo', 'final')
    checkpoint = grpo_final if os.path.isdir(grpo_final) else SFT_SAVE_PATH

    if not os.path.isdir(checkpoint):
        log(f"⚠️ No checkpoint found at {checkpoint} — skipping push.")
        return

    log(f"Loading checkpoint from {checkpoint} for hub push ...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        checkpoint, device_map='cpu', trust_remote_code=True
    )

    hub_id = "samarthdave0305/cascadeguard-trained"
    log(f"Pushing to {hub_id} ...")
    model.push_to_hub(hub_id, token=HF_TOKEN)
    tokenizer.push_to_hub(hub_id, token=HF_TOKEN)
    log(f"✅ Model pushed to https://huggingface.co/{hub_id}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN training thread
# ═══════════════════════════════════════════════════════════════════════════════
def run_training():
    global training_done
    try:
        log("🚀 CascadeGuard training pipeline starting ...")
        verify_gpu()
        install_deps()
        clone_repo()
        ensure_output_dirs()
        install_package()
        prewarm_osm()
        tokenizer, model, sft_raw = load_dataset_and_model()
        #sft_records  = run_sft(tokenizer, model, sft_raw)

        #del model
        #if torch.cuda.is_available():
         #   torch.cuda.empty_cache()

        grpo_records = run_grpo()
        plot_curves(sft_records, grpo_records)
        push_to_hub()
        log("=" * 60)
        log("🎉 ALL STEPS COMPLETE! Training finished successfully.")
        log("=" * 60)
    except Exception as e:
        import traceback
        log(f"❌ Training failed: {e}")
        log(traceback.format_exc())
    finally:
        training_done = True

training_thread = threading.Thread(target=run_training, daemon=True)
training_thread.start()

# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════

def get_logs():
    status = "✅ COMPLETE" if training_done else "🔄 RUNNING..."
    return f"Status: {status}\n\n" + "\n".join(log_lines[-150:])

def get_loss_curve():
    path = os.path.join(CURVES_DIR, 'training_loss_curve.png')
    if os.path.exists(path):
        return path
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, 'Loss curve will appear here once SFT starts',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=12, color='gray')
    ax.set_axis_off()
    tmp = os.path.join('/tmp', 'placeholder_loss.png')
    fig.savefig(tmp, dpi=100, bbox_inches='tight')
    plt.close()
    return tmp

def get_reward_curve():
    path = os.path.join(CURVES_DIR, 'training_reward_curve.png')
    if os.path.exists(path):
        return path
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, 'Reward curve will appear here once GRPO starts',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=12, color='gray')
    ax.set_axis_off()
    tmp = os.path.join('/tmp', 'placeholder_reward.png')
    fig.savefig(tmp, dpi=100, bbox_inches='tight')
    plt.close()
    return tmp

def get_metrics():
    lines = []
    status = "COMPLETE ✅" if training_done else "RUNNING 🔄"
    lines.append(f"Status      : {status}")
    lines.append(f"Log lines   : {len(log_lines)}")

    if sft_records:
        losses = [r['loss'] for r in sft_records if 'loss' in r and not math.isnan(r['loss'])]
        lrs    = [r['lr']   for r in sft_records if 'lr'   in r]
        lines.append("")
        lines.append("─── SFT ───────────────────────────────")
        lines.append(f"Steps completed : {sft_records[-1]['step']}")
        if losses:
            lines.append(f"Initial loss    : {losses[0]:.4f}")
            lines.append(f"Final loss      : {losses[-1]:.4f}")
            lines.append(f"Min loss        : {min(losses):.4f}")
        if lrs:
            lines.append(f"Final LR        : {lrs[-1]:.2e}")
    else:
        lines.append("")
        lines.append("─── SFT ─── (not started yet)")

    if grpo_records:
        rewards = [r['reward'] for r in grpo_records if 'reward' in r]
        lines.append("")
        lines.append("─── GRPO ──────────────────────────────")
        lines.append(f"Steps logged    : {len(grpo_records)}")
        if rewards:
            lines.append(f"Initial reward  : {rewards[0]:.4f}")
            lines.append(f"Final reward    : {rewards[-1]:.4f}")
            lines.append(f"Max reward      : {max(rewards):.4f}")
            lines.append(f"Mean reward     : {sum(rewards)/len(rewards):.4f}")
    else:
        lines.append("")
        lines.append("─── GRPO ─── (not started yet)")

    if torch.cuda.is_available():
        lines.append("")
        lines.append("─── GPU ───────────────────────────────")
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        lines.append(f"VRAM used   : {used:.1f} / {total:.1f} GB")
        lines.append(f"GPU name    : {torch.cuda.get_device_properties(0).name}")

    return "\n".join(lines)

def refresh_all():
    return get_logs(), get_loss_curve(), get_reward_curve(), get_metrics()

with gr.Blocks(title="CascadeGuard Training") as demo:
    gr.Markdown(
        "# 🛡️ CascadeGuard — SFT + GRPO Training\n"
        "Training runs automatically on Space start. "
        "Panels refresh every 15 s — or click **Refresh** manually."
    )

    with gr.Tabs():
        with gr.TabItem("📋 Live Logs"):
            log_box = gr.Textbox(
                label="Training Logs",
                lines=35, max_lines=35,
                interactive=False,
            )

        with gr.TabItem("📈 Training Curves"):
            with gr.Row():
                loss_img   = gr.Image(label="Loss Curve (SFT + GRPO)",  type="filepath")
                reward_img = gr.Image(label="Reward Curve (GRPO)",       type="filepath")

        with gr.TabItem("📊 Metrics"):
            metrics_box = gr.Textbox(
                label="Training Metrics",
                lines=25, max_lines=25,
                interactive=False,
            )

    refresh_btn = gr.Button("🔄 Refresh All", variant="primary")
    refresh_btn.click(
        fn=refresh_all,
        outputs=[log_box, loss_img, reward_img, metrics_box],
    )

    demo.load(
        fn=refresh_all,
        outputs=[log_box, loss_img, reward_img, metrics_box],
    )

    timer = gr.Timer(value=15)
    timer.tick(
        fn=refresh_all,
        outputs=[log_box, loss_img, reward_img, metrics_box],
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    theme=gr.themes.Soft(),
)
