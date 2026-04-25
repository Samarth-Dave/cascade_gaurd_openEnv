from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Real-world node metadata — Mumbai Metropolitan Area (OSM-verified)
# Injected into every task config at module load time.
# Format: node_id → (lat, lon, service_radius_km, real_name)
# ---------------------------------------------------------------------------
_NODE_REAL_DATA: Dict[str, Tuple[float, float, float, str]] = {
    "POWER_GEN_1":      (19.0553, 72.9036, 50.0, "Tata Power Trombay Plant"),
    "POWER_GEN_2":      (19.2200, 72.9780, 45.0, "Bhiwandi Power Plant"),
    "POWER_GEN_3":      (19.3000, 73.0500, 45.0, "Kalyan Generating Station"),
    "POWER_TRANS_1":    (19.1200, 72.8400, 30.0, "Bandra Transmission Hub"),
    "POWER_TRANS_2":    (19.1800, 72.9600, 30.0, "Thane Transmission Hub"),
    "POWER_TRANS_3":    (19.2500, 73.0000, 28.0, "Dombivli Transmission Hub"),
    "POWER_DIST_1":     (19.0900, 72.8600, 20.0, "Dadar Distribution Centre"),
    "POWER_DIST_2":     (19.1500, 72.9000, 20.0, "Kurla Distribution Centre"),
    "WATER_TREAT_1":    (19.0700, 72.8300, 25.0, "Worli Water Treatment Plant"),
    "WATER_TREAT_2":    (19.2000, 72.8500, 22.0, "Bhandup Treatment Works"),
    "WATER_PUMP_1":     (19.0600, 72.8200, 15.0, "Lower Parel Pump Station"),
    "WATER_PUMP_2":     (19.1050, 72.8700, 15.0, "Matunga Pump Station"),
    "WATER_PUMP_3":     (19.1900, 72.8400, 14.0, "Mulund Pump Station"),
    "WATER_DIST_1":     (19.0800, 72.8550, 12.0, "Sion Distribution Node"),
    "WATER_DIST_2":     (19.2100, 72.8600, 11.0, "Vikhroli Distribution Node"),
    "HOSP_1":           (19.0012, 72.8402,  8.0, "KEM Hospital Parel"),
    "HOSP_2":           (19.0406, 72.8696,  8.0, "Sion Municipal Hospital"),
    "HOSP_3":           (19.1600, 72.8200,  8.0, "Rajawadi General Hospital"),
    "EMERG_1":          (19.0019, 72.8337, 10.0, "Nair Hospital"),
    "EMERG_2":          (19.2300, 72.9500,  9.0, "Thane Civil Hospital"),
    "TELECOM_1":        (19.1000, 72.8200, 100.0, "Bandra Telecom Tower"),
    "TELECOM_2":        (19.0500, 72.9000, 100.0, "South Mumbai Relay Tower"),
    "TELECOM_SWITCH_1": (19.0800, 72.8700,  80.0, "Central Telecom Exchange"),
    "TELECOM_3":        (19.3200, 73.0800,  95.0, "Kalyan Eastern Mast"),
}


def _enrich_task_nodes(cfg: dict) -> None:
    """
    Inject real lat/lon, service_radius_km, and real_name into every node
    that exists in _NODE_REAL_DATA. Existing lat/lon are overwritten with
    accurate real-world coordinates. Preserves all other node fields.
    """
    for n in cfg.get("nodes", []):
        nid = n["node_id"]
        if nid not in _NODE_REAL_DATA:
            continue
        lat, lon, radius, name = _NODE_REAL_DATA[nid]
        n["lat"] = lat
        n["lon"] = lon
        n.setdefault("service_radius_km", radius)
        n["real_name"] = name

TASK_CONFIGS: Dict[str, dict] = {
    "task_easy": {
        "task_id": "task_easy",
        "seed": 42,
        "max_steps": 10,
        "budget": 5.0,
        "nodes": [
            {"node_id": "POWER_GEN_1",   "sector": "power", "is_critical": False, "lat": 19.076, "lon": 72.877, "service_radius_km": 50.0},
            {"node_id": "POWER_TRANS_1", "sector": "power", "is_critical": False, "lat": 19.120, "lon": 72.840, "service_radius_km": 30.0},
            {"node_id": "POWER_TRANS_2", "sector": "power", "is_critical": False, "lat": 19.180, "lon": 72.960, "service_radius_km": 30.0},
            {"node_id": "POWER_DIST_1",  "sector": "power", "is_critical": False, "lat": 19.090, "lon": 72.860, "service_radius_km": 20.0},
            {"node_id": "POWER_DIST_2",  "sector": "power", "is_critical": False, "lat": 19.150, "lon": 72.900, "service_radius_km": 20.0},
            {"node_id": "HOSP_1",        "sector": "hospital", "is_critical": True, "lat": 19.0012, "lon": 72.8402, "service_radius_km": 8.0},
        ],
        "edges": [
            {"source_id": "POWER_GEN_1",   "target_id": "POWER_TRANS_1"},
            {"source_id": "POWER_GEN_1",   "target_id": "POWER_TRANS_2"},
            {"source_id": "POWER_TRANS_1", "target_id": "POWER_DIST_1"},
            {"source_id": "POWER_TRANS_2", "target_id": "POWER_DIST_2"},
            {"source_id": "POWER_DIST_1",  "target_id": "HOSP_1"},
        ],
        "stress_schedule": {
            2: {"type": "equipment_fault", "target": "POWER_DIST_1", "effect": -0.75},
        },
        "delayed_sectors": [],
        "partial_obs_nodes": [],
        "description": "Power grid with one critical hospital dependency. Equipment fault hits POWER_DIST_1 at step 2.",
    },

    "task_medium": {
        "task_id": "task_medium",
        "seed": 123,
        "max_steps": 20,
        "budget": 10.0,
        "nodes": [
            # power (5)
            {"node_id": "POWER_GEN_1",   "sector": "power",    "is_critical": False, "lat": 19.076, "lon": 72.877, "service_radius_km": 50.0},
            {"node_id": "POWER_TRANS_1", "sector": "power",    "is_critical": False, "lat": 19.120, "lon": 72.840, "service_radius_km": 30.0},
            {"node_id": "POWER_TRANS_2", "sector": "power",    "is_critical": False, "lat": 19.180, "lon": 72.960, "service_radius_km": 30.0},
            {"node_id": "POWER_DIST_1",  "sector": "power",    "is_critical": False, "lat": 19.090, "lon": 72.860, "service_radius_km": 20.0},
            {"node_id": "POWER_DIST_2",  "sector": "power",    "is_critical": False, "lat": 19.150, "lon": 72.900, "service_radius_km": 20.0},
            # water (4)
            {"node_id": "WATER_TREAT_1", "sector": "water",    "is_critical": False, "lat": 19.070, "lon": 72.830, "service_radius_km": 25.0},
            {"node_id": "WATER_PUMP_1",  "sector": "water",    "is_critical": False, "lat": 19.060, "lon": 72.820, "service_radius_km": 15.0},
            {"node_id": "WATER_PUMP_2",  "sector": "water",    "is_critical": False, "lat": 19.105, "lon": 72.870, "service_radius_km": 15.0},
            {"node_id": "WATER_DIST_1",  "sector": "water",    "is_critical": False, "lat": 19.080, "lon": 72.855, "service_radius_km": 12.0},
            # hospital (3)
            {"node_id": "HOSP_1",        "sector": "hospital", "is_critical": True,  "lat": 19.055, "lon": 72.835, "service_radius_km": 8.0},
            {"node_id": "HOSP_2",        "sector": "hospital", "is_critical": True,  "lat": 19.115, "lon": 72.875, "service_radius_km": 8.0},
            {"node_id": "EMERG_1",       "sector": "hospital", "is_critical": False, "lat": 19.075, "lon": 72.845, "service_radius_km": 10.0},
        ],
        "edges": [
            # power backbone
            {"source_id": "POWER_GEN_1",   "target_id": "POWER_TRANS_1"},
            {"source_id": "POWER_GEN_1",   "target_id": "POWER_TRANS_2"},
            {"source_id": "POWER_TRANS_1", "target_id": "POWER_DIST_1"},
            {"source_id": "POWER_TRANS_2", "target_id": "POWER_DIST_2"},
            # water -> power
            {"source_id": "POWER_DIST_1",  "target_id": "WATER_TREAT_1"},
            {"source_id": "POWER_DIST_1",  "target_id": "WATER_PUMP_1"},
            {"source_id": "POWER_DIST_2",  "target_id": "WATER_PUMP_2"},
            {"source_id": "WATER_PUMP_1",  "target_id": "WATER_DIST_1"},
            {"source_id": "WATER_PUMP_2",  "target_id": "WATER_DIST_1"},
            # hospital -> power + water
            {"source_id": "POWER_DIST_1",  "target_id": "HOSP_1"},
            {"source_id": "WATER_DIST_1",  "target_id": "HOSP_1"},
            {"source_id": "POWER_DIST_2",  "target_id": "HOSP_2"},
            {"source_id": "WATER_DIST_1",  "target_id": "HOSP_2"},
            {"source_id": "POWER_DIST_1",  "target_id": "EMERG_1"},
        ],
        "stress_schedule": {
            4:  {"type": "weather", "target": None, "effect": "storm_warning"},
            8:  {"type": "weather", "target": None, "effect": "extreme_storm"},
            11: {"type": "weather", "target": None, "effect": "clear"},
        },
        "delayed_sectors": ["water"],
        "partial_obs_nodes": [],
        "description": (
            "Power + Water + Hospital triad with incoming storm. "
            "storm_warning at step 4, extreme_storm at step 8 for 4 steps. "
            "Water nodes have 2-step observation delay."
        ),
    },

    "task_hard": {
        "task_id": "task_hard",
        "seed": 999,
        "max_steps": 30,
        "budget": 15.0,
        "nodes": [
            # power (5)
            {"node_id": "POWER_GEN_1",      "sector": "power",    "is_critical": False},
            {"node_id": "POWER_TRANS_1",    "sector": "power",    "is_critical": False},
            {"node_id": "POWER_TRANS_2",    "sector": "power",    "is_critical": False},
            {"node_id": "POWER_DIST_1",     "sector": "power",    "is_critical": False},
            {"node_id": "POWER_DIST_2",     "sector": "power",    "is_critical": False},
            # water (4)
            {"node_id": "WATER_TREAT_1",    "sector": "water",    "is_critical": False},
            {"node_id": "WATER_PUMP_1",     "sector": "water",    "is_critical": False},
            {"node_id": "WATER_PUMP_2",     "sector": "water",    "is_critical": False},
            {"node_id": "WATER_DIST_1",     "sector": "water",    "is_critical": False},
            # hospital (3)
            {"node_id": "HOSP_1",           "sector": "hospital", "is_critical": True},
            {"node_id": "HOSP_2",           "sector": "hospital", "is_critical": True},
            {"node_id": "EMERG_1",          "sector": "hospital", "is_critical": False},
            # telecom (3)
            {"node_id": "TELECOM_1",        "sector": "telecom",  "is_critical": False},
            {"node_id": "TELECOM_2",        "sector": "telecom",  "is_critical": False},
            {"node_id": "TELECOM_SWITCH_1", "sector": "telecom",  "is_critical": False},
        ],
        "edges": [
            # power backbone
            {"source_id": "POWER_GEN_1",      "target_id": "POWER_TRANS_1"},
            {"source_id": "POWER_GEN_1",      "target_id": "POWER_TRANS_2"},
            {"source_id": "POWER_TRANS_1",    "target_id": "POWER_DIST_1"},
            {"source_id": "POWER_TRANS_2",    "target_id": "POWER_DIST_2"},
            # water -> power
            {"source_id": "POWER_DIST_1",     "target_id": "WATER_TREAT_1"},
            {"source_id": "POWER_DIST_1",     "target_id": "WATER_PUMP_1"},
            {"source_id": "POWER_DIST_2",     "target_id": "WATER_PUMP_2"},
            {"source_id": "WATER_PUMP_1",     "target_id": "WATER_DIST_1"},
            {"source_id": "WATER_PUMP_2",     "target_id": "WATER_DIST_1"},
            # hospital -> power + water
            {"source_id": "POWER_DIST_1",     "target_id": "HOSP_1"},
            {"source_id": "WATER_DIST_1",     "target_id": "HOSP_1"},
            {"source_id": "POWER_DIST_2",     "target_id": "HOSP_2"},
            {"source_id": "WATER_DIST_1",     "target_id": "HOSP_2"},
            {"source_id": "POWER_DIST_1",     "target_id": "EMERG_1"},
            # telecom -> power
            {"source_id": "POWER_TRANS_1",    "target_id": "TELECOM_1"},
            {"source_id": "POWER_TRANS_2",    "target_id": "TELECOM_2"},
            {"source_id": "TELECOM_1",        "target_id": "TELECOM_SWITCH_1"},
            {"source_id": "TELECOM_2",        "target_id": "TELECOM_SWITCH_1"},
        ],
        "stress_schedule": {
            # SCADA anomaly from step 1 onward, recurring -0.05/step
            1:  {"type": "scada_anomaly", "target": "TELECOM_SWITCH_1", "effect": -0.02, "recurring": True},
            5:  {"type": "weather",       "target": None,               "effect": "extreme_storm"},
            10: {"type": "weather",       "target": None,               "effect": "clear"},
        },
        "delayed_sectors": [],
        "partial_obs_nodes": [
            "WATER_PUMP_1",
            "WATER_PUMP_2",
            "TELECOM_1",
            "TELECOM_2",
        ],
        "description": (
            "Full 4-sector crisis with partial observability. "
            "SCADA anomaly on TELECOM_SWITCH_1 from step 1 (-0.05/step). "
            "Extreme storm at step 5, clears at step 15. "
            "6 nodes always delayed. Tight budget requiring triage."
        ),
    },

    # ------------------------------------------------------------------
    # Task 4: Root generator blackout - tests graph-centrality reasoning
    # ------------------------------------------------------------------
    "task_gen_blackout": {
        "task_id": "task_gen_blackout",
        "seed": 777,
        "max_steps": 15,
        "budget": 8.0,
        "nodes": [
            # power (5)
            {"node_id": "POWER_GEN_1",   "sector": "power",    "is_critical": False},
            {"node_id": "POWER_TRANS_1", "sector": "power",    "is_critical": False},
            {"node_id": "POWER_TRANS_2", "sector": "power",    "is_critical": False},
            {"node_id": "POWER_DIST_1",  "sector": "power",    "is_critical": False},
            {"node_id": "POWER_DIST_2",  "sector": "power",    "is_critical": False},
            # hospital (3) - directly depend on power distribution
            {"node_id": "HOSP_1",        "sector": "hospital", "is_critical": True},
            {"node_id": "HOSP_2",        "sector": "hospital", "is_critical": True},
            {"node_id": "EMERG_1",       "sector": "hospital", "is_critical": False},
        ],
        "edges": [
            {"source_id": "POWER_GEN_1",   "target_id": "POWER_TRANS_1"},
            {"source_id": "POWER_GEN_1",   "target_id": "POWER_TRANS_2"},
            {"source_id": "POWER_TRANS_1", "target_id": "POWER_DIST_1"},
            {"source_id": "POWER_TRANS_2", "target_id": "POWER_DIST_2"},
            # Hospitals depend directly on power distribution
            {"source_id": "POWER_DIST_1",  "target_id": "HOSP_1"},
            {"source_id": "POWER_DIST_2",  "target_id": "HOSP_2"},
            {"source_id": "POWER_DIST_1",  "target_id": "EMERG_1"},
        ],
        "stress_schedule": {
            # Root node equipment fault - if not hardened, full TRANS->DIST->HOSP cascade
            3: {"type": "equipment_fault", "target": "POWER_GEN_1",  "effect": -0.85},
            # Second fault: a distribution node while still recovering root
            8: {"type": "equipment_fault", "target": "POWER_DIST_2", "effect": -0.80},
        },
        "delayed_sectors": [],
        "partial_obs_nodes": [],
        "description": (
            "Root generator failure cascade scenario. POWER_GEN_1 suffers major fault at "
            "step 3 (-0.85), which cascades through TRANS->DIST->HOSPITAL chain over "
            "multiple steps. Second POWER_DIST_2 fault at step 8 threatens HOSP_2. "
            "Agent must harden the ROOT node and critical hospitals BEFORE the cascade. "
            "Tests whether agent understands graph centrality vs. leaf vulnerability."
        ),
    },

    # ------------------------------------------------------------------
    # Task 5: Cyber-physical attack - persistent SCADA + multi-fault
    # ------------------------------------------------------------------
    "task_cyberattack": {
        "task_id": "task_cyberattack",
        "seed": 314,
        "max_steps": 25,
        "budget": 13.0,
        "nodes": [
            # power (5)
            {"node_id": "POWER_GEN_1",      "sector": "power",    "is_critical": False},
            {"node_id": "POWER_TRANS_1",    "sector": "power",    "is_critical": False},
            {"node_id": "POWER_TRANS_2",    "sector": "power",    "is_critical": False},
            {"node_id": "POWER_DIST_1",     "sector": "power",    "is_critical": False},
            {"node_id": "POWER_DIST_2",     "sector": "power",    "is_critical": False},
            # hospital (3)
            {"node_id": "HOSP_1",           "sector": "hospital", "is_critical": True},
            {"node_id": "HOSP_2",           "sector": "hospital", "is_critical": True},
            {"node_id": "EMERG_1",          "sector": "hospital", "is_critical": False},
            # telecom (3)
            {"node_id": "TELECOM_1",        "sector": "telecom",  "is_critical": False},
            {"node_id": "TELECOM_2",        "sector": "telecom",  "is_critical": False},
            {"node_id": "TELECOM_SWITCH_1", "sector": "telecom",  "is_critical": False},
        ],
        "edges": [
            {"source_id": "POWER_GEN_1",      "target_id": "POWER_TRANS_1"},
            {"source_id": "POWER_GEN_1",      "target_id": "POWER_TRANS_2"},
            {"source_id": "POWER_TRANS_1",    "target_id": "POWER_DIST_1"},
            {"source_id": "POWER_TRANS_2",    "target_id": "POWER_DIST_2"},
            {"source_id": "POWER_DIST_1",     "target_id": "HOSP_1"},
            {"source_id": "POWER_DIST_2",     "target_id": "HOSP_2"},
            {"source_id": "POWER_DIST_1",     "target_id": "EMERG_1"},
            # Telecom depends on power transmission
            {"source_id": "POWER_TRANS_1",    "target_id": "TELECOM_1"},
            {"source_id": "POWER_TRANS_2",    "target_id": "TELECOM_2"},
            {"source_id": "TELECOM_1",        "target_id": "TELECOM_SWITCH_1"},
            {"source_id": "TELECOM_2",        "target_id": "TELECOM_SWITCH_1"},
        ],
        "stress_schedule": {
            # Persistent SCADA cyberattack: telecom switch degrades every step
            1:  {"type": "scada_anomaly", "target": "TELECOM_SWITCH_1", "effect": -0.03, "recurring": True},
            # Equipment fault on POWER_DIST_1 -> threatens HOSP_1 + EMERG_1
            4:  {"type": "equipment_fault", "target": "POWER_DIST_1", "effect": -0.65},
            # Storm compounds the damage
            10: {"type": "weather",         "target": None, "effect": "storm_warning"},
            14: {"type": "weather",         "target": None, "effect": "extreme_storm"},
            17: {"type": "weather",         "target": None, "effect": "clear"},
            # Late fault when agent may be resource-exhausted
            20: {"type": "equipment_fault", "target": "POWER_DIST_2", "effect": -0.60},
        },
        "delayed_sectors": [],
        "partial_obs_nodes": ["TELECOM_1", "TELECOM_2", "HOSP_2"],
        "description": (
            "Cyber-physical compound attack. Persistent SCADA anomaly on TELECOM_SWITCH_1 "
            "from step 1 (-0.05/step). POWER_DIST_1 equipment fault at step 4 threatening "
            "HOSP_1 and EMERG_1. Storm warning at step 10, extreme storm at step 14. "
            "Second power fault at step 20 when resources may be depleted. "
            "3 nodes always delayed (TELECOM_1, TELECOM_2, HOSP_2). "
            "Tests: sustained threat management, hospital protection, budget triage."
        ),
    },

    # ------------------------------------------------------------------
    # Task 6: Surge-demand stress test - no direct equipment faults
    # ------------------------------------------------------------------
    "task_surge_demand": {
        "task_id": "task_surge_demand",
        "seed": 888,
        "max_steps": 18,
        "budget": 9.0,
        "nodes": [
            # power (5)
            {"node_id": "POWER_GEN_1",   "sector": "power",    "is_critical": False, "lat": 19.076, "lon": 72.877, "service_radius_km": 50.0},
            {"node_id": "POWER_TRANS_1", "sector": "power",    "is_critical": False, "lat": 19.120, "lon": 72.840, "service_radius_km": 30.0},
            {"node_id": "POWER_TRANS_2", "sector": "power",    "is_critical": False, "lat": 19.180, "lon": 72.960, "service_radius_km": 30.0},
            {"node_id": "POWER_DIST_1",  "sector": "power",    "is_critical": False, "lat": 19.090, "lon": 72.860, "service_radius_km": 20.0},
            {"node_id": "POWER_DIST_2",  "sector": "power",    "is_critical": False, "lat": 19.150, "lon": 72.900, "service_radius_km": 20.0},
            # water (4)
            {"node_id": "WATER_TREAT_1", "sector": "water",    "is_critical": False, "lat": 19.070, "lon": 72.830, "service_radius_km": 25.0},
            {"node_id": "WATER_PUMP_1",  "sector": "water",    "is_critical": False, "lat": 19.060, "lon": 72.820, "service_radius_km": 15.0},
            {"node_id": "WATER_PUMP_2",  "sector": "water",    "is_critical": False, "lat": 19.105, "lon": 72.870, "service_radius_km": 15.0},
            {"node_id": "WATER_DIST_1",  "sector": "water",    "is_critical": False, "lat": 19.080, "lon": 72.855, "service_radius_km": 12.0},
            # hospital (3)
            {"node_id": "HOSP_1",        "sector": "hospital", "is_critical": True,  "lat": 19.055, "lon": 72.835, "service_radius_km": 8.0},
            {"node_id": "HOSP_2",        "sector": "hospital", "is_critical": True,  "lat": 19.115, "lon": 72.875, "service_radius_km": 8.0},
            {"node_id": "EMERG_1",       "sector": "hospital", "is_critical": False, "lat": 19.075, "lon": 72.845, "service_radius_km": 10.0},
        ],
        "edges": [
            # power backbone
            {"source_id": "POWER_GEN_1",   "target_id": "POWER_TRANS_1"},
            {"source_id": "POWER_GEN_1",   "target_id": "POWER_TRANS_2"},
            {"source_id": "POWER_TRANS_1", "target_id": "POWER_DIST_1"},
            {"source_id": "POWER_TRANS_2", "target_id": "POWER_DIST_2"},
            # water -> power
            {"source_id": "POWER_DIST_1",  "target_id": "WATER_TREAT_1"},
            {"source_id": "POWER_DIST_1",  "target_id": "WATER_PUMP_1"},
            {"source_id": "POWER_DIST_2",  "target_id": "WATER_PUMP_2"},
            {"source_id": "WATER_PUMP_1",  "target_id": "WATER_DIST_1"},
            {"source_id": "WATER_PUMP_2",  "target_id": "WATER_DIST_1"},
            # hospital -> power + water
            {"source_id": "POWER_DIST_1",  "target_id": "HOSP_1"},
            {"source_id": "WATER_DIST_1",  "target_id": "HOSP_1"},
            {"source_id": "POWER_DIST_2",  "target_id": "HOSP_2"},
            {"source_id": "WATER_DIST_1",  "target_id": "HOSP_2"},
            {"source_id": "POWER_DIST_1",  "target_id": "EMERG_1"},
        ],
        "stress_schedule": {
            3:  {"type": "load_surge", "target": None, "effect": 0.20},
            8:  {"type": "load_surge", "target": None, "effect": 0.20},
            14: {"type": "load_surge", "target": None, "effect": 0.20},
        },
        "delayed_sectors": ["water"],
        "partial_obs_nodes": [],
        "description": (
            "Demand-surge resilience scenario with no direct equipment faults. "
            "System load spikes by +20% at steps 3, 8, and 14; success depends on "
            "timely shed_load and coordination decisions under moderate budget pressure."
        ),
    },

    # ------------------------------------------------------------------
    # Task 7: Metro City — Geo-realistic full city resilience scenario
    # 18 nodes, real Mumbai-inspired coordinates, monsoon story arc
    # ------------------------------------------------------------------
    "task_real_city": {
        "task_id": "task_real_city",
        "seed": 2026,
        "max_steps": 35,
        "budget": 15.0,
        "nodes": [
            # Power Generation (3) — large radius, wide coverage
            {"node_id": "POWER_GEN_1",   "sector": "power",    "is_critical": False, "lat": 19.076, "lon": 72.877, "service_radius_km": 50.0},
            {"node_id": "POWER_GEN_2",   "sector": "power",    "is_critical": False, "lat": 19.220, "lon": 72.978, "service_radius_km": 45.0},
            {"node_id": "POWER_GEN_3",   "sector": "power",    "is_critical": False, "lat": 19.300, "lon": 73.050, "service_radius_km": 45.0},
            # Transmission (3) — mid radius
            {"node_id": "POWER_TRANS_1", "sector": "power",    "is_critical": False, "lat": 19.120, "lon": 72.840, "service_radius_km": 30.0},
            {"node_id": "POWER_TRANS_2", "sector": "power",    "is_critical": False, "lat": 19.180, "lon": 72.960, "service_radius_km": 30.0},
            {"node_id": "POWER_TRANS_3", "sector": "power",    "is_critical": False, "lat": 19.250, "lon": 73.000, "service_radius_km": 28.0},
            # Distribution (3) — local radius
            {"node_id": "POWER_DIST_1",  "sector": "power",    "is_critical": False, "lat": 19.090, "lon": 72.860, "service_radius_km": 20.0},
            {"node_id": "POWER_DIST_2",  "sector": "power",    "is_critical": False, "lat": 19.150, "lon": 72.900, "service_radius_km": 20.0},
            # Water (3)
            {"node_id": "WATER_TREAT_1", "sector": "water",    "is_critical": False, "lat": 19.070, "lon": 72.830, "service_radius_km": 25.0},
            {"node_id": "WATER_TREAT_2", "sector": "water",    "is_critical": False, "lat": 19.200, "lon": 72.850, "service_radius_km": 22.0},
            {"node_id": "WATER_PUMP_1",  "sector": "water",    "is_critical": False, "lat": 19.060, "lon": 72.820, "service_radius_km": 15.0},
            {"node_id": "WATER_DIST_1",  "sector": "water",    "is_critical": False, "lat": 19.080, "lon": 72.855, "service_radius_km": 12.0},
            # Hospital (4) — critical care nodes
            {"node_id": "HOSP_1",        "sector": "hospital", "is_critical": True,  "lat": 19.055, "lon": 72.835, "service_radius_km": 8.0},
            {"node_id": "HOSP_2",        "sector": "hospital", "is_critical": True,  "lat": 19.115, "lon": 72.875, "service_radius_km": 8.0},
            {"node_id": "HOSP_3",        "sector": "hospital", "is_critical": True,  "lat": 19.160, "lon": 72.820, "service_radius_km": 8.0},
            {"node_id": "EMERG_1",       "sector": "hospital", "is_critical": False, "lat": 19.075, "lon": 72.845, "service_radius_km": 10.0},
            # Telecom (3) — large radius, covers nodes for observation confidence
            {"node_id": "TELECOM_1",        "sector": "telecom",  "is_critical": False, "lat": 19.100, "lon": 72.820, "service_radius_km": 100.0},
            {"node_id": "TELECOM_2",        "sector": "telecom",  "is_critical": False, "lat": 19.050, "lon": 72.900, "service_radius_km": 100.0},
        ],
        "edges": [
            # Power backbone
            {"source_id": "POWER_GEN_1",   "target_id": "POWER_TRANS_1"},
            {"source_id": "POWER_GEN_1",   "target_id": "POWER_TRANS_2"},
            {"source_id": "POWER_GEN_2",   "target_id": "POWER_TRANS_2"},
            {"source_id": "POWER_GEN_2",   "target_id": "POWER_TRANS_3"},
            {"source_id": "POWER_GEN_3",   "target_id": "POWER_TRANS_3"},
            {"source_id": "POWER_TRANS_1", "target_id": "POWER_DIST_1"},
            {"source_id": "POWER_TRANS_2", "target_id": "POWER_DIST_2"},
            # Water -> power
            {"source_id": "POWER_DIST_1",  "target_id": "WATER_TREAT_1"},
            {"source_id": "POWER_DIST_2",  "target_id": "WATER_TREAT_2"},
            {"source_id": "POWER_DIST_1",  "target_id": "WATER_PUMP_1"},
            {"source_id": "WATER_PUMP_1",  "target_id": "WATER_DIST_1"},
            # Hospital -> power + water
            {"source_id": "POWER_DIST_1",  "target_id": "HOSP_1"},
            {"source_id": "WATER_DIST_1",  "target_id": "HOSP_1"},
            {"source_id": "POWER_DIST_2",  "target_id": "HOSP_2"},
            {"source_id": "WATER_TREAT_2", "target_id": "HOSP_2"},
            {"source_id": "POWER_DIST_2",  "target_id": "HOSP_3"},
            {"source_id": "POWER_DIST_1",  "target_id": "EMERG_1"},
            # Telecom -> power transmission
            {"source_id": "POWER_TRANS_1", "target_id": "TELECOM_1"},
            {"source_id": "POWER_TRANS_2", "target_id": "TELECOM_2"},
        ],
        "stress_schedule": {
            # Monsoon phase 1: storm warning
            3:  {"type": "weather",       "target": None,            "effect": "storm_warning"},
            # Equipment fault during storm warning — tests proactive hardening
            5:  {"type": "equipment_fault","target": "POWER_GEN_1",  "effect": -0.60},
            # Storm escalates
            8:  {"type": "weather",       "target": None,            "effect": "extreme_storm"},
            # Second fault — cascades to hospitals
            12: {"type": "equipment_fault","target": "POWER_DIST_1", "effect": -0.70},
            # SCADA anomaly on telecom — degrades observability
            15: {"type": "scada_anomaly", "target": "TELECOM_1",    "effect": -0.04, "recurring": True},
            # Storm clears — recovery window
            20: {"type": "weather",       "target": None,            "effect": "clear"},
            # Late fault — tests if agent saved budget
            28: {"type": "equipment_fault","target": "POWER_GEN_2",  "effect": -0.55},
        },
        "delayed_sectors": ["water"],
        "partial_obs_nodes": ["TELECOM_1", "HOSP_3"],
        "description": (
            "Metro City Resilience — 18-node Mumbai-inspired city with real geo-coordinates. "
            "Monsoon season: storm_warning at step 3, POWER_GEN_1 fault step 5, extreme_storm step 8, "
            "POWER_DIST_1 cascade fault step 12 threatening hospitals, "
            "SCADA anomaly on TELECOM_1 from step 15 (degrades observability), "
            "storm clears step 20, late POWER_GEN_2 fault step 28. "
            "Tests geo-aware radius coverage: POWER_GEN_2 (45km) partially covers HOSP_2 area. "
            "Agent must prioritise backbone nodes, manage telecom for observation confidence, "
            "and protect 4 hospital nodes across a 35-step monsoon crisis."
        ),
    },
}


TASK_SEED_SPLITS: Dict[str, Dict[str, List[int]]] = {
    "task_easy": {
        "train": [42, 1042, 2042, 3042, 4042],
        "validation": [5042, 6042],
        "holdout": [7042, 8042],
    },
    "task_medium": {
        "train": [123, 1123, 2123, 3123, 4123],
        "validation": [5123, 6123],
        "holdout": [7123, 8123],
    },
    "task_hard": {
        "train": [999, 1999, 2999, 3999, 4999],
        "validation": [5999, 6999],
        "holdout": [7999, 8999],
    },
    "task_gen_blackout": {
        "train": [777, 1777, 2777, 3777, 4777],
        "validation": [5777, 6777],
        "holdout": [7777, 8777],
    },
    "task_cyberattack": {
        "train": [314, 1314, 2314, 3314, 4314],
        "validation": [5314, 6314],
        "holdout": [7314, 8314],
    },
    "task_surge_demand": {
        "train": [888, 1888, 2888, 3888, 4888],
        "validation": [5888, 6888],
        "holdout": [7888, 8888],
    },
    "task_real_city": {
        "train": [2026, 3026, 4026, 5026, 6026],
        "validation": [7026, 8026],
        "holdout": [9026, 10026],
    },
}


def resolve_seed(task_id: str, split: str = "train", scenario_index: int = 0, fallback_seed: int | None = None) -> int:
    if task_id not in TASK_CONFIGS:
        raise KeyError(f"Unknown task_id {task_id!r}. Valid options: {list(TASK_CONFIGS)}")

    split_map = TASK_SEED_SPLITS.get(task_id, {})
    seed_list = split_map.get(split)
    if seed_list:
        return seed_list[scenario_index % len(seed_list)]

    if fallback_seed is not None:
        return fallback_seed
    return int(TASK_CONFIGS[task_id].get("seed", 0))


def _pick_same_sector_target(cfg: dict, target_node: str, rng: random.Random) -> str:
    sector_by_node = {n["node_id"]: n["sector"] for n in cfg.get("nodes", [])}
    if target_node not in sector_by_node:
        return target_node

    sector = sector_by_node[target_node]
    candidates = [
        n["node_id"]
        for n in cfg.get("nodes", [])
        if n["sector"] == sector and n["node_id"] != target_node
    ]
    if not candidates:
        return target_node
    return rng.choice(candidates)


def _jitter_schedule(cfg: dict, rng: random.Random) -> dict:
    base = cfg.get("stress_schedule", {})
    max_steps = int(cfg.get("max_steps", 10))
    min_step = 1
    max_step = max(1, max_steps - 2)

    reserved: set[int] = set()
    out: dict[int, dict] = {}

    for step, event in sorted(base.items(), key=lambda x: x[0]):
        jitter = rng.randint(-1, 1)
        candidate = max(min_step, min(max_step, step + jitter))
        while candidate in reserved:
            candidate = min(max_step, candidate + 1)
            if candidate in reserved:
                candidate = max(min_step, candidate - 2)
            if candidate not in reserved:
                break

        ev = copy.deepcopy(event)
        if ev.get("type") == "equipment_fault" and ev.get("target") and rng.random() < 0.35:
            ev["target"] = _pick_same_sector_target(cfg, ev["target"], rng)

        out[candidate] = ev
        reserved.add(candidate)

    return dict(sorted(out.items(), key=lambda x: x[0]))


def _sample_observability(cfg: dict, rng: random.Random) -> None:
    if cfg.get("task_id") == "task_medium" and rng.random() < 0.4:
        delayed = set(cfg.get("delayed_sectors", []))
        delayed.add("water")
        if rng.random() < 0.10:
            delayed.add("hospital")
        cfg["delayed_sectors"] = sorted(delayed)

    if cfg.get("task_id") == "task_hard":
        # Keep hard-task observability deterministic to reduce high-variance seeds.
        return

    if cfg.get("task_id") == "task_cyberattack":
        eligible = [n["node_id"] for n in cfg.get("nodes", []) if not n.get("is_critical", False)]
        if eligible:
            k = min(len(eligible), max(2, int(round(0.2 * len(eligible)))))
            cfg["partial_obs_nodes"] = sorted(rng.sample(eligible, k=k))


def materialize_task_config(task_id: str, seed: int, split: str = "train") -> dict:
    """Return a deterministic per-seed scenario instance from a task template."""
    if task_id not in TASK_CONFIGS:
        raise KeyError(f"Unknown task_id {task_id!r}. Valid options: {list(TASK_CONFIGS)}")

    cfg = copy.deepcopy(TASK_CONFIGS[task_id])
    cfg["seed"] = int(seed)
    cfg["scenario_split"] = split

    rng = random.Random(int(seed) + (0 if split == "train" else 10_000 if split == "validation" else 20_000))
    cfg["stress_schedule"] = _jitter_schedule(cfg, rng)
    _sample_observability(cfg, rng)
    return cfg


def get_task(task_id: str) -> dict:
    """Return the task config dict for the given task_id.

    Raises KeyError if task_id is not found.
    """
    if task_id not in TASK_CONFIGS:
        raise KeyError(
            f"Unknown task_id {task_id!r}. Valid options: {list(TASK_CONFIGS)}"
        )
    return copy.deepcopy(TASK_CONFIGS[task_id])


# ---------------------------------------------------------------------------
# Step 1 — Enrich ALL existing task nodes with real-world coordinates
# ---------------------------------------------------------------------------
for _cfg in TASK_CONFIGS.values():
    _enrich_task_nodes(_cfg)


# ---------------------------------------------------------------------------
# Step 2 — Build OSM city task configs from data/real_nodes.json
#           Falls back gracefully if the file doesn't exist yet.
# ---------------------------------------------------------------------------

# Stress schedules per city task (same cascade physics, different topology)
_OSM_STRESS_SCHEDULES: Dict[str, dict] = {
    "london": {
        3:  {"type": "equipment_fault", "target": None,  "effect": -0.65},
        8:  {"type": "weather",         "target": None,  "effect": "storm_warning"},
        12: {"type": "weather",         "target": None,  "effect": "extreme_storm"},
        18: {"type": "weather",         "target": None,  "effect": "clear"},
        22: {"type": "equipment_fault", "target": None,  "effect": -0.55},
    },
    "mumbai": {
        3:  {"type": "weather",         "target": None,  "effect": "storm_warning"},
        5:  {"type": "equipment_fault", "target": None,  "effect": -0.60},
        8:  {"type": "weather",         "target": None,  "effect": "extreme_storm"},
        12: {"type": "equipment_fault", "target": None,  "effect": -0.70},
        20: {"type": "weather",         "target": None,  "effect": "clear"},
        28: {"type": "equipment_fault", "target": None,  "effect": -0.55},
    },
    "bangalore": {
        4:  {"type": "equipment_fault", "target": None,  "effect": -0.60},
        10: {"type": "weather",         "target": None,  "effect": "storm_warning"},
        15: {"type": "equipment_fault", "target": None,  "effect": -0.65},
        20: {"type": "weather",         "target": None,  "effect": "clear"},
    },
    "delhi": {
        2:  {"type": "equipment_fault", "target": None,  "effect": -0.55},
        7:  {"type": "weather",         "target": None,  "effect": "storm_warning"},
        12: {"type": "weather",         "target": None,  "effect": "extreme_storm"},
        16: {"type": "weather",         "target": None,  "effect": "clear"},
        22: {"type": "equipment_fault", "target": None,  "effect": -0.60},
    },
    "nyc": {
        3:  {"type": "equipment_fault", "target": None,  "effect": -0.65},
        6:  {"type": "weather",         "target": None,  "effect": "storm_warning"},
        10: {"type": "weather",         "target": None,  "effect": "extreme_storm"},
        16: {"type": "weather",         "target": None,  "effect": "clear"},
        24: {"type": "equipment_fault", "target": None,  "effect": -0.60},
    },
    "tokyo": {
        2:  {"type": "equipment_fault", "target": None,  "effect": -0.70},
        5:  {"type": "weather",         "target": None,  "effect": "storm_warning"},
        9:  {"type": "weather",         "target": None,  "effect": "extreme_storm"},
        14: {"type": "weather",         "target": None,  "effect": "clear"},
        20: {"type": "equipment_fault", "target": None,  "effect": -0.55},
    },
}

_OSM_SEEDS = {
    "london":    4242,
    "mumbai":    4343,
    "bangalore": 4444,
    "delhi":     4545,
    "nyc":       4646,
    "tokyo":     4747,
}


def _build_osm_task(city: str, city_data: dict) -> dict:
    """
    Convert a city entry from real_nodes.json into a TASK_CONFIGS-compatible dict.
    The first power node found in the stress schedule's "target": None slots
    is resolved dynamically in materialize_task_config.
    """
    raw_nodes = city_data.get("nodes", [])
    raw_edges = city_data.get("edges", [])

    # Identify the first power node to use as default stress target
    power_nodes = [n["id"] for n in raw_nodes if n.get("sector") == "power"]
    hosp_nodes  = [n["id"] for n in raw_nodes if n.get("sector") == "hospital"]
    default_power = power_nodes[0] if power_nodes else (raw_nodes[0]["id"] if raw_nodes else "UNKNOWN")

    # Build node configs — is_critical = True for hospital nodes
    nodes = []
    for n in raw_nodes:
        is_critical = n.get("sector") == "hospital"
        nodes.append({
            "node_id":           n["id"],
            "sector":            n.get("sector", "power"),
            "is_critical":       is_critical,
            "lat":               n.get("lat", 0.0),
            "lon":               n.get("lng", 0.0),   # tasks.py uses "lon"
            "service_radius_km": n.get("service_radius_km", 15.0),
            "real_name":         n.get("name", n["id"]),
        })

    # Convert edges
    edges = [{"source_id": e[0], "target_id": e[1]} for e in raw_edges
             if len(e) == 2]

    # Resolve "target": None → first power node in stress schedule
    stress = copy.deepcopy(_OSM_STRESS_SCHEDULES.get(city, {}))
    for event in stress.values():
        if event.get("target") is None and event.get("type") == "equipment_fault":
            event["target"] = default_power

    # Partial obs: non-critical telecom nodes
    telecom_ids = [n["id"] for n in raw_nodes if n.get("sector") == "telecom"]
    partial_obs = telecom_ids[:2]

    return {
        "task_id":         f"task_osm_{city}",
        "seed":            _OSM_SEEDS[city],
        "max_steps":       30,
        "budget":          15.0,
        "nodes":           nodes,
        "edges":           edges,
        "stress_schedule": stress,
        "delayed_sectors": [],
        "partial_obs_nodes": partial_obs,
        "city":            city,
        "description": (
            f"Real-world {city.title()} infrastructure resilience. "
            f"{len(nodes)} real OSM nodes ({len(power_nodes)} power, "
            f"{len(hosp_nodes)} hospitals). "
            f"Node positions sourced from OpenStreetMap Overpass API."
        ),
    }


def _load_osm_tasks() -> None:
    """Load real_nodes.json and register task_osm_<city> in TASK_CONFIGS."""
    data_path = Path(__file__).parent / "data" / "real_nodes.json"
    if not data_path.exists():
        return  # Fetcher hasn't run yet — silently skip

    try:
        city_data = json.loads(data_path.read_text(encoding="utf-8"))
    except Exception:
        return

    for city, data in city_data.items():
        task_id = f"task_osm_{city}"
        try:
            cfg = _build_osm_task(city, data)
            TASK_CONFIGS[task_id] = cfg
            TASK_SEED_SPLITS[task_id] = {
                "train":      [_OSM_SEEDS[city] + i * 100 for i in range(5)],
                "validation": [_OSM_SEEDS[city] + 500, _OSM_SEEDS[city] + 600],
                "holdout":    [_OSM_SEEDS[city] + 700, _OSM_SEEDS[city] + 800],
            }
        except Exception:
            pass  # Malformed city data — skip silently


# Register OSM tasks at import time
_load_osm_tasks()
