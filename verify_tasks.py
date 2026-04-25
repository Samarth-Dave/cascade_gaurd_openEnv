import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = REPO_ROOT.parent

try:
    from cascade_guard.server.cascade_environment import CascadeEnvironment
    from cascade_guard.models import CascadeAction
    from cascade_guard.server.graders import grade
    from cascade_guard.inference import _build_user_prompt, SYSTEM_PROMPT, smart_heuristic
except ModuleNotFoundError:
    if str(PACKAGE_PARENT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_PARENT))
    try:
        from cascade_guard.server.cascade_environment import CascadeEnvironment
        from cascade_guard.models import CascadeAction
        from cascade_guard.server.graders import grade
        from cascade_guard.inference import _build_user_prompt, SYSTEM_PROMPT, smart_heuristic
    except ModuleNotFoundError:
        # Legacy repo-root execution fallback.
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from server.cascade_environment import CascadeEnvironment
        from models import CascadeAction
        from server.graders import grade
        from inference import _build_user_prompt, SYSTEM_PROMPT, smart_heuristic

for tid in [
    "task_easy",
    "task_medium",
    "task_hard",
    "task_gen_blackout",
    "task_cyberattack",
    "task_surge_demand",
]:
    env = CascadeEnvironment()
    obs = env.reset(task_id=tid)
    print(f"{tid}: nodes={len(obs.nodes)} edges={len(obs.edges)} budget={obs.budget_remaining} steps={obs.max_steps}")

env = CascadeEnvironment()
obs = env.reset(task_id="task_hard")
prompt = _build_user_prompt(obs, 0)
tok_est = (len(SYSTEM_PROMPT) + len(prompt)) // 4
print(f"task_hard clean prompt: chars={len(prompt)}, total_est_tokens~{tok_est}")
print(f"Prompt preview: {repr(prompt[:300])}")

s1 = grade(
    "task_gen_blackout",
    failure_history=[],
    hospital_health_log=[1.0, 1.0],
    final_sector_summary={"power": 0.8, "hospital": 0.9},
    budget_spent=2.0,
    budget_total=8.0,
)
s2 = grade(
    "task_cyberattack",
    failure_history=[],
    hospital_health_log=[1.0] * 5,
    final_sector_summary={"power": 0.7, "hospital": 0.9, "telecom": 0.6},
    budget_spent=4.0,
    budget_total=10.0,
)
s3 = grade(
    "task_surge_demand",
    failure_history=[],
    hospital_health_log=[1.0] * 5,
    final_sector_summary={"power": 0.85, "water": 0.8, "hospital": 0.9},
    budget_spent=3.0,
    budget_total=9.0,
)
print(f"grade_gen_blackout={s1}  grade_cyberattack={s2}  grade_surge_demand={s3}")

env2 = CascadeEnvironment()
obs2 = env2.reset(task_id="task_easy")
h = smart_heuristic(obs2, step=0)
print(f"heuristic step=0 task_easy: {h}")

env2.step(CascadeAction(action_type="wait", target_node_id=None))
env2.step(CascadeAction(action_type="wait", target_node_id=None))
obs3 = env2.step(CascadeAction(action_type="wait", target_node_id=None))
h2 = smart_heuristic(obs3, step=3)
print(f"heuristic step=3 task_easy (after fault): {h2}")

print("ALL OK")
