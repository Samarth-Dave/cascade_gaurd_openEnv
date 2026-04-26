import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_PARENT = REPO_ROOT.parent

try:
    from cascade_guard.tasks import TASK_CONFIGS
    from cascade_guard.reward import compute_reward, grpo_verifier
except ModuleNotFoundError:
    if str(PACKAGE_PARENT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_PARENT))
    try:
        from cascade_guard.tasks import TASK_CONFIGS
        from cascade_guard.reward import compute_reward, grpo_verifier
    except ModuleNotFoundError:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from tasks import TASK_CONFIGS
        from reward import compute_reward, grpo_verifier

osm = [t for t in TASK_CONFIGS if t.startswith("task_osm")]
base = [t for t in TASK_CONFIGS if not t.startswith("task_osm")]

print("Base tasks:", len(base))
for t in base:
    nodes = TASK_CONFIGS[t].get("nodes", [])
    enriched = sum(1 for n in nodes if "real_name" in n)
    has_coord = sum(1 for n in nodes if n.get("lat", 0) != 0)
    print(f"  {t}: {len(nodes)} nodes, {enriched} real_names, {has_coord} with coords")

print()
print("OSM tasks:", len(osm))
for t in osm:
    nodes = TASK_CONFIGS[t].get("nodes", [])
    city = TASK_CONFIGS[t].get("city", "?")
    n0 = nodes[0]
    print(f"  {t}: {city}, {len(nodes)} nodes, sample node={n0['node_id']} ({n0.get('real_name','?')}) @ {n0.get('lat')},{n0.get('lon')}")

print()
print("reward.py import check:", end=" ")
print("OK")

print()
print("ALL CHECKS PASSED")
