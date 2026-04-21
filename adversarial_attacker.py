"""
adversarial_attacker.py — Adversarial attacker agent for CascadeGuard multi-agent mode
========================================================================================
Implements three attacker strategies:

  1. random      — attacks a random non-hardened node every N steps
  2. centrality  — targets the highest-degree node (backbone proxy)
  3. hospital    — targets power suppliers of hospital nodes

The attacker integrates with CascadeEnvironment via:
    env.inject_attack(attack_event)   # call BEFORE env.step(defender_action)

This transforms CascadeGuard into a two-player zero-sum game:
  - Defender (LLM agent): recovers/hardens/sheds load to maintain health
  - Attacker (adversarial policy): degrades infrastructure to force collapse

This directly unlocks OpenEnv Theme 1 (Multi-Agent) and the Fleet AI sub-theme.

Usage:
    from cascade_guard.adversarial_attacker import make_attacker

    attacker = make_attacker("centrality", attack_strength=0.12, attack_every=3)
    obs = env.reset(task_id="task_hard", seed=seed)
    done = False
    while not done:
        defender_action = get_llm_action(obs)
        attack = attacker.select_attack(obs, step=obs.step)
        env.inject_attack(attack)
        obs = env.step(defender_action)
        done = obs.done
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional


# ── Base class ─────────────────────────────────────────────────────────────────

class BaseAttacker:
    """
    Abstract base class for all attacker strategies.

    Subclasses implement select_attack() to return an attack event dict
    (compatible with CascadeEnvironment._apply_stress_event) or None.
    """

    def __init__(self, attack_strength: float = 0.12) -> None:
        self.attack_strength = attack_strength
        self.rng = random.Random(12345)  # reproducible by default

    def select_attack(self, obs, step: int) -> Optional[Dict]:
        """
        Compute the attacker's move for this step.

        Args:
            obs:  Current CascadeObservation (from env.reset or env.step).
            step: Current step number (use obs.step for consistency).

        Returns:
            Attack event dict suitable for env.inject_attack(), or None.
            Format: {"type": "equipment_fault", "target": NODE_ID, "effect": -float}
        """
        raise NotImplementedError

    # ── Helper selectors ──────────────────────────────────────────────────────

    def _non_hardened_operational_nodes(self, obs) -> List:
        """All nodes that are operational and not hardened (vulnerable targets)."""
        return [n for n in obs.nodes if n.is_operational and not n.is_hardened]

    def _hospital_power_suppliers(self, obs) -> List:
        """
        Power-sector nodes that are direct suppliers of hospital nodes.
        Attacking these maximises hospital damage.
        """
        hosp_ids = {n.node_id for n in obs.nodes if n.sector == "hospital"}
        supplier_ids: set[str] = set()
        for e in obs.edges:
            if e.target_id in hosp_ids:
                supplier_ids.add(e.source_id)
        return [
            n for n in obs.nodes
            if n.node_id in supplier_ids
            and n.is_operational
            and n.sector == "power"
        ]

    def _make_event(self, target_node_id: str) -> Dict:
        """Build a standard equipment_fault event dict."""
        return {
            "type":   "equipment_fault",
            "target": target_node_id,
            "effect": -self.attack_strength,
        }


# ── Strategy 1: Random attacker ────────────────────────────────────────────────

class RandomAttacker(BaseAttacker):
    """
    Selects a random non-hardened operational node every `attack_every` steps.
    The simplest baseline to test defender resilience.
    """

    def __init__(
        self, attack_strength: float = 0.12, attack_every: int = 3
    ) -> None:
        super().__init__(attack_strength)
        self.attack_every = attack_every

    def select_attack(self, obs, step: int) -> Optional[Dict]:
        if step % self.attack_every != 0:
            return None
        candidates = self._non_hardened_operational_nodes(obs)
        if not candidates:
            return None
        target = self.rng.choice(candidates)
        return self._make_event(target.node_id)


# ── Strategy 2: Centrality attacker ───────────────────────────────────────────

class CentralityAttacker(BaseAttacker):
    """
    Targets the highest-degree node as a proxy for betweenness centrality.

    Degree = number of edges (both upstream + downstream) incident to a node.
    High-degree nodes are backbone infrastructure — knocking them out causes
    the widest cascade, forcing the defender to protect them proactively.
    """

    def __init__(
        self, attack_strength: float = 0.15, attack_every: int = 4
    ) -> None:
        super().__init__(attack_strength)
        self.attack_every = attack_every

    def _degree_map(self, obs) -> Dict[str, int]:
        degree: Dict[str, int] = {n.node_id: 0 for n in obs.nodes}
        for e in obs.edges:
            degree[e.source_id] = degree.get(e.source_id, 0) + 1
            degree[e.target_id] = degree.get(e.target_id, 0) + 1
        return degree

    def select_attack(self, obs, step: int) -> Optional[Dict]:
        if step % self.attack_every != 0:
            return None
        candidates = self._non_hardened_operational_nodes(obs)
        if not candidates:
            return None
        degree = self._degree_map(obs)
        # Target highest-degree vulnerable node
        target = max(candidates, key=lambda n: degree.get(n.node_id, 0))
        return self._make_event(target.node_id)


# ── Strategy 3: Hospital-targeting attacker ────────────────────────────────────

class HospitalTargetingAttacker(BaseAttacker):
    """
    Specifically targets power suppliers of hospital nodes.

    This is the hardest strategy for the defender: it maximises hospital
    health degradation and triggers the heaviest reward penalties.
    Falls back to the highest-load operational node if no hospital suppliers
    are available (e.g., all have been hardened).
    """

    def __init__(
        self, attack_strength: float = 0.18, attack_every: int = 5
    ) -> None:
        super().__init__(attack_strength)
        self.attack_every = attack_every

    def select_attack(self, obs, step: int) -> Optional[Dict]:
        if step % self.attack_every != 0:
            return None

        candidates = self._hospital_power_suppliers(obs)

        if not candidates:
            # Fallback: highest-load operational node (most likely to cascade)
            candidates = [
                n for n in obs.nodes
                if n.is_operational and not n.is_hardened and n.load > 0.7
            ]

        if not candidates:
            # Last resort: any non-hardened operational node
            candidates = self._non_hardened_operational_nodes(obs)

        if not candidates:
            return None

        target = self.rng.choice(candidates)
        return self._make_event(target.node_id)


# ── Registry & factory ─────────────────────────────────────────────────────────

ATTACKER_REGISTRY: Dict[str, type] = {
    "random":     RandomAttacker,
    "centrality": CentralityAttacker,
    "hospital":   HospitalTargetingAttacker,
}


def make_attacker(name: str = "centrality", **kwargs) -> BaseAttacker:
    """
    Factory function for attacker strategies.

    Args:
        name:    One of "random", "centrality", "hospital".
        **kwargs: Passed to the attacker constructor
                  (attack_strength, attack_every).

    Returns:
        Instantiated attacker.

    Raises:
        KeyError: If name is not in ATTACKER_REGISTRY.
    """
    if name not in ATTACKER_REGISTRY:
        raise KeyError(
            f"Unknown attacker {name!r}. Valid options: {list(ATTACKER_REGISTRY)}"
        )
    return ATTACKER_REGISTRY[name](**kwargs)
