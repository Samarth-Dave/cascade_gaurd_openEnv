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
    "random":      RandomAttacker,
    "centrality":  CentralityAttacker,
    "hospital":    HospitalTargetingAttacker,
    "betweenness": None,  # set below after class definition
}


# ── Strategy 4: Real Betweenness Centrality Attacker ──────────────────────────

class BetweennessCentralityAttacker(BaseAttacker):
    """
    Adversarial attacker that computes **real betweenness centrality** from the
    live observation graph and targets the highest-centrality healthy node.

    Betweenness centrality measures how often a node lies on the shortest path
    between every other pair of nodes — high-betweenness nodes are the true
    chokepoints of the infrastructure graph.

    Unlike CentralityAttacker (degree proxy), this uses BFS across all node
    pairs to compute exact normalized betweenness — identical to the formula
    in CascadeEnvironment._compute_centrality().

    Additional behaviours
    ---------------------
    - Backs off when cascade depth >= 2 (let existing cascade propagate).
    - Escalates attack strength over time (step / 100 bonus).
    - Only attacks nodes above a minimum centrality threshold (0.05).
    - Exposes .think(obs, step) for UI chain-of-thought display.
    """

    def __init__(
        self,
        attack_strength: float = 0.15,
        attack_every: int = 3,
        min_centrality: float = 0.05,
    ) -> None:
        super().__init__(attack_strength)
        self.attack_every     = attack_every
        self.min_centrality   = min_centrality
        self._history: List[str] = []   # previously attacked node IDs

    # ── Betweenness centrality (BFS-based, same as env) ───────────────────────

    def _compute_betweenness(self, obs) -> Dict[str, float]:
        """Approximate betweenness centrality from the live observation graph."""
        node_ids  = [n.node_id for n in obs.nodes]
        adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        for e in obs.edges:
            if e.source_id in adj and e.target_id in adj:
                adj[e.source_id].append(e.target_id)
                adj[e.target_id].append(e.source_id)

        centrality: Dict[str, float] = {nid: 0.0 for nid in node_ids}
        n = len(node_ids)

        for source in node_ids:
            # BFS from source
            dist: Dict[str, int]       = {source: 0}
            num_paths: Dict[str, int]  = {nid: 0 for nid in node_ids}
            num_paths[source]          = 1
            preds: Dict[str, List[str]] = {nid: [] for nid in node_ids}
            queue: List[str]            = [source]

            while queue:
                v = queue.pop(0)
                for w in adj[v]:
                    if w not in dist:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist.get(w, -1) == dist[v] + 1:
                        num_paths[w] += num_paths[v]
                        preds[w].append(v)

            dep: Dict[str, float] = {nid: 0.0 for nid in node_ids}
            for w in sorted(dist, key=lambda x: -dist[x]):
                for v in preds[w]:
                    if num_paths[w] > 0:
                        dep[v] += (num_paths[v] / num_paths[w]) * (1 + dep[w])
                if w != source:
                    centrality[w] += dep[w]

        norm = max(1, (n - 1) * (n - 2))
        return {nid: round(c / norm, 4) for nid, c in centrality.items()}

    # ── Attack selection ───────────────────────────────────────────────────────

    def select_attack(self, obs, step: int) -> Optional[Dict]:
        # Back off if cascade is already spreading
        cascade_depth = sum(
            1 for n in obs.nodes if not n.is_operational
        )
        if cascade_depth >= 2:
            return None

        if step % self.attack_every != 0:
            return None

        centrality = self._compute_betweenness(obs)

        candidates = [
            n for n in obs.nodes
            if n.is_operational
            and not n.is_hardened
            and n.node_id not in self._history
            and centrality.get(n.node_id, 0.0) >= self.min_centrality
        ]

        if not candidates:
            # Retry without history filter
            candidates = [
                n for n in obs.nodes
                if n.is_operational and not n.is_hardened
                and centrality.get(n.node_id, 0.0) >= self.min_centrality
            ]

        if not candidates:
            return None

        target = max(candidates, key=lambda n: centrality.get(n.node_id, 0.0))
        self._history.append(target.node_id)

        escalated = self.attack_strength + (step / 100.0)
        return {
            "type":    "equipment_fault",
            "target":  target.node_id,
            "effect":  -round(min(0.85, escalated), 3),
        }

    # ── Chain-of-thought for UI display ───────────────────────────────────────

    def think(self, obs, step: int) -> str:
        """
        Returns attacker chain-of-thought text for the UI sidebar.
        Shows real betweenness rankings and planned action.
        """
        centrality  = self._compute_betweenness(obs)
        top_targets = sorted(
            [(n.node_id, centrality.get(n.node_id, 0.0), n.sector)
             for n in obs.nodes if n.is_operational],
            key=lambda x: -x[1],
        )[:4]

        cascade_depth = sum(1 for n in obs.nodes if not n.is_operational)
        budget_info   = getattr(obs, "budget_remaining", "?")

        lines = [
            f"[ATTACKER] Step {step} — Real betweenness centrality analysis",
            f"Top targets: " + ", ".join(
                f"{nid}={c:.3f}({s})" for nid, c, s in top_targets
            ),
            f"Cascade depth: {cascade_depth} | Defender budget: {budget_info}",
        ]

        attack = self.select_attack(obs, step)
        if attack:
            strength = abs(attack["effect"])
            lines.append(
                f"ACTION: INJECT {attack['target']} "
                f"strength={strength:.3f} "
                f"centrality={centrality.get(attack['target'], 0):.3f}"
            )
        else:
            if cascade_depth >= 2:
                lines.append("ACTION: HOLD — cascade already propagating (depth≥2)")
            else:
                lines.append("ACTION: HOLD — no viable high-centrality target")

        return "\n".join(lines)


# Update registry with betweenness attacker
ATTACKER_REGISTRY["betweenness"] = BetweennessCentralityAttacker


def make_attacker(name: str = "betweenness", **kwargs) -> BaseAttacker:
    """
    Factory function for attacker strategies.

    Args:
        name:    One of "random", "centrality", "hospital", "betweenness".
        **kwargs: Passed to the attacker constructor.

    Returns:
        Instantiated attacker.
    """
    if name not in ATTACKER_REGISTRY:
        raise KeyError(
            f"Unknown attacker {name!r}. Valid options: {list(ATTACKER_REGISTRY)}"
        )
    return ATTACKER_REGISTRY[name](**kwargs)
