# ROOT CAUSE FIXES - Phase 2 Submission Failure Analysis & Resolution

## Problem Summary
All 10 Phase 2 submissions failed despite comprehensive code fixes. Root cause analysis identified **three systemic issues** that explain the failures.

## Root Cause #1: Missing Score Field in CascadeState

**Issue:** The `CascadeState` model in `models.py` was missing a `score` field. The `state()` method in `cascade_environment.py` never computed the episode score, so when the validator called `env.state().score`, it received either `None`, `0.0` (default), or an `AttributeError`.

**Fix Applied:**
- Added `score: float = 0.5` field to `CascadeState` class in `models.py` (line ~105)
- Updated `cascade_environment.py` to:
  - Import the `grade` function from `cascade_guard.server.graders`
  - Compute episode score in `state()` method by calling the grader with all necessary data
  - Return the computed score in the `CascadeState` object

**Files Modified:**
- [models.py](models.py#L105) - Added score field
- [server/cascade_environment.py](server/cascade_environment.py#L24) - Added grader import
- [server/cascade_environment.py](server/cascade_environment.py#L556) - Updated state() method

## Root Cause #2: EVAL_SPLIT Mismatch in Baseline Mode

**Issue:** The baseline evaluation mode (the default when `EVAL_MODE` is not set) hardcoded `scenario_split="train"` but the environment variable `EVAL_SPLIT` defaults to `"holdout"`. When the validator re-ran evaluation, it would use different seeds than the JSON report showed, causing seed mismatches.

**Specifics:**
- Default `EVAL_SPLIT = "holdout"` (line 70)
- But baseline mode used `seed_list = TASK_SEED_SPLITS[task_id]["train"]`
- This inconsistency caused seeds to be completely different on validator re-runs

**Fix Applied:**
- Changed baseline mode to use `EVAL_SPLIT` consistently:
  - Line 1699: `"train"` → `EVAL_SPLIT`
  - Line 1710: `scenario_split="train"` → `scenario_split=EVAL_SPLIT`
  - Line 1717: `"scenario_split": "train"` → `"scenario_split": EVAL_SPLIT`

**Files Modified:**
- [inference.py](inference.py#L1699) - Fixed seed retrieval
- [inference.py](inference.py#L1710) - Fixed scenario_split parameter
- [inference.py](inference.py#L1717) - Fixed JSON logging

## Root Cause #3: Correct Files Already in Place

**Verification:** The question arose whether the graders were being read from the wrong location. However, verification confirmed:

✅ `inference.py` correctly imports from `cascade_guard.server.graders` (line 26)
✅ `cascade_guard/server/graders.py` exists and contains all safety fixes
✅ No root-level file confusion

## Verification Results

```
[SUCCESS] All three files compile without syntax errors
[SUCCESS] verify_tasks.py: ALL OK
[SUCCESS] All test scenarios pass
[SUCCESS] Score guarantee: (0.0001, 0.9999) maintained
```

## Expected Phase 2 Outcome

With these fixes:

1. ✅ **Score field accessible** - Validator can now read `state().score` and get the correct computed score
2. ✅ **Consistent evaluation** - Baseline mode uses EVAL_SPLIT correctly, no seed mismatches
3. ✅ **Proper grading pipeline** - Episode scores computed via the multi-layer safety pipeline
4. ✅ **Boundary safety maintained** - All scores strictly in (0.0001, 0.9999) range

## Timeline

| Phase | Status |
|-------|--------|
| Fix 1: Add score field to CascadeState | ✅ Complete |
| Fix 2: Update state() method to compute score | ✅ Complete |
| Fix 3: Fix EVAL_SPLIT in baseline mode | ✅ Complete |
| Syntax validation | ✅ Complete |
| Integration testing | ✅ Complete |
| **Ready for Phase 2 Submission** | ✅ **YES** |

## Deployment Instructions

To deploy these fixes:

```bash
# Commit changes to repo
git add -A
git commit -m "Fix: Add score field, compute episode score, fix EVAL_SPLIT mismatch"
git push

# Redeploy HF Space with updated code
# (The validator will pull and run the latest code)
```

---

**Date Fixed:** April 8, 2026
**Root Cause Discovery:** Systemic analysis revealed all three issues were preventing validator from scoring submissions
**Expected Result:** Phase 2 validation should now pass with correct scores in (0.0001, 0.9999) range
