"""
Final end-to-end verification for training readiness.
Tests: real_name through full pipeline, curriculum, prompt, reward.
"""
import sys

PASS = []
FAIL = []

def check(name, fn):
    try:
        result = fn()
        PASS.append(f"  PASS  {name}" + (f": {result}" if result else ""))
    except Exception as e:
        FAIL.append(f"  FAIL  {name}: {e}")

# 1. real_name flows from tasks.py -> env -> NodeStatus
def test_real_name_pipeline():
    from cascade_guard.server.cascade_environment import CascadeEnvironment
    env = CascadeEnvironment()
    obs = env.reset(task_id="task_hard", seed=42)
    names = {n.node_id: n.real_name for n in obs.nodes}
    assert names.get("HOSP_1") == "KEM Hospital Parel", f"Got: {names.get('HOSP_1')}"
    assert names.get("POWER_GEN_1") == "Tata Power Trombay Plant", f"Got: {names.get('POWER_GEN_1')}"
    return f"HOSP_1='{names['HOSP_1']}'"

check("real_name in task_hard (HOSP_1)", test_real_name_pipeline)

# 2. OSM tasks have real_name in observations
def test_osm_real_name():
    from cascade_guard.server.cascade_environment import CascadeEnvironment
    env = CascadeEnvironment()
    obs = env.reset(task_id="task_osm_london", seed=42)
    names = {n.node_id: n.real_name for n in obs.nodes}
    hosp = [(nid, name) for nid, name in names.items() if "Hospital" in name]
    assert len(hosp) > 0, f"No hospitals found. names={names}"
    return f"{len(hosp)} hospitals: {hosp[0]}"

check("real_name in task_osm_london", test_osm_real_name)

# 3. Prompt builder shows real_name column
def test_prompt_real_name():
    from cascade_guard.server.cascade_environment import CascadeEnvironment
    from cascade_guard.training.cot_prompt import build_user_prompt
    env = CascadeEnvironment()
    obs = env.reset(task_id="task_osm_mumbai", seed=42)
    prompt = build_user_prompt(obs)
    assert "Real Name" in prompt, "Missing 'Real Name' column header"
    assert "KEM Hospital" in prompt or "Sion" in prompt or "Nair" in prompt, \
        "No real hospital name in prompt"
    assert "Tata Power" in prompt or "Dharavi" in prompt or "Bandra" in prompt, \
        "No real power node name in prompt"
    return "Real names visible in LLM prompt"

check("real_name in LLM prompt (task_osm_mumbai)", test_prompt_real_name)

# 4. Curriculum loads all 11 stages (5 base + 6 OSM)
def test_curriculum():
    from cascade_guard.training.curriculum_scheduler import CurriculumScheduler
    sched = CurriculumScheduler()
    stages = [s.task_id for s in sched._stages]
    osm = [s for s in stages if s.startswith("task_osm")]
    base = [s for s in stages if not s.startswith("task_osm")]
    assert len(base) == 5, f"Expected 5 base stages, got {len(base)}: {base}"
    assert len(osm) == 6, f"Expected 6 OSM stages, got {len(osm)}: {osm}"
    assert sched.current_task == "task_easy"
    return f"{len(stages)} total stages (5 base + 6 OSM)"

check("CurriculumScheduler has 11 stages", test_curriculum)

# 5. reward.py grpo_verifier works
def test_reward():
    from cascade_guard.reward import grpo_verifier, parse_action_from_completion
    obs = {
        "sector_health": {"power": 0.9, "water": 0.8, "hospital": 1.0, "telecom": 0.7},
        "cascade_depth": 0,
        "budget": 12.0,
        "node_states": {
            "HOSP_1": {"status": "operational", "centrality": 0.4},
            "POWER_GEN_1": {"status": "operational", "centrality": 0.6},
        }
    }
    r = grpo_verifier("<think>All good</think><action>wait(null)</action>", obs, {"action_type": "wait"})
    assert -1.0 <= r <= 1.0, f"Reward out of range: {r}"
    action = parse_action_from_completion("<action>recover(HOSP_1)</action>")
    assert action["action_type"] == "recover"
    assert action["target_node_id"] == "HOSP_1"
    return f"reward={r}, parsed action={action['action_type']}({action['target_node_id']})"

check("reward.py grpo_verifier + parse_action", test_reward)

# 6. adversarial BetweennessCentralityAttacker + .think()
def test_attacker():
    from cascade_guard.adversarial_attacker import make_attacker
    from cascade_guard.server.cascade_environment import CascadeEnvironment
    env = CascadeEnvironment()
    obs = env.reset(task_id="task_osm_london", seed=42)
    attacker = make_attacker("betweenness")
    thought = attacker.think(obs, step=3)
    assert "betweenness" in thought.lower()
    assert "Top targets" in thought
    return f"think() OK, {len(thought)} chars"

check("BetweennessCentralityAttacker.think()", test_attacker)

# 7. Full step loop on OSM task
def test_osm_step_loop():
    from cascade_guard.server.cascade_environment import CascadeEnvironment
    from cascade_guard.models import CascadeAction
    env = CascadeEnvironment()
    obs = env.reset(task_id="task_osm_tokyo", seed=42)
    rewards = []
    for _ in range(5):
        obs = env.step(CascadeAction(action_type="wait"))
        rewards.append(obs.reward)
    assert len(rewards) == 5
    state = env.state
    assert 0.0 < state.score < 1.0
    return f"5 steps OK, rewards={[round(r,2) for r in rewards]}, score={state.score:.3f}"

check("5-step loop on task_osm_tokyo", test_osm_step_loop)

# 8. /cities endpoint data consistency
def test_cities_data():
    import json
    from pathlib import Path
    data = json.loads((Path("data/real_nodes.json")).read_text())
    for city, v in data.items():
        nodes = v.get("nodes", [])
        hosps = [n for n in nodes if n.get("sector") == "hospital"]
        power = [n for n in nodes if n.get("sector") == "power"]
        assert len(hosps) >= 2, f"{city}: only {len(hosps)} hospitals"
        assert len(power) >= 2, f"{city}: only {len(power)} power nodes"
        assert all(n.get("name") for n in nodes), f"{city}: some nodes missing name"
    return f"{len(data)} cities, all have >=2 hospitals and >=2 power nodes"

check("real_nodes.json completeness", test_cities_data)

# ── Results ────────────────────────────────────────────────────────────────────
print()
for line in PASS:
    print(line)
if FAIL:
    print()
    for line in FAIL:
        print(line)
    print(f"\n{len(FAIL)} checks FAILED")
    sys.exit(1)
else:
    print(f"\nAll {len(PASS)} checks PASSED. Training pipeline is production-ready.")
