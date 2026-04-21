"""
trajectory_logger.py — Log full agent trajectories for before/after comparison
================================================================================
Records step-by-step observations, actions, rewards, and grader scores.
Produces per-episode JSON artifacts and a human-readable markdown comparison.

Usage:
    logger = TrajectoryLogger("logs/trajectories")
    ep_id = logger.start_episode(task_id="task_easy", model_label="random_baseline")
    for each step:
        logger.record_step(obs, "recover", "POWER_DIST_1", reward, info_dict)
    logger.end_episode(grader_score=0.42)
    logger.write_comparison_report("logs/comparison.md")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    step: int
    action: str
    target: Optional[str]
    reward: float
    newly_failed: List[str]
    completed_recoveries: List[str]
    sector_health: Dict[str, float]
    hospital_min_health: float
    budget_remaining: float
    weather: str
    action_valid: bool
    reasoning: str = ""   # LLM's <think> block if captured (truncated to 500 chars)


@dataclass
class EpisodeRecord:
    episode_id: str
    task_id: str
    model_label: str
    seed: int
    grader_score: float
    steps: List[StepRecord] = field(default_factory=list)
    started_at: str = ""
    ended_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── Logger class ───────────────────────────────────────────────────────────────

class TrajectoryLogger:
    """
    Records full agent trajectories for qualitative before/after comparison.

    One logger instance can hold multiple episodes across multiple tasks and
    model variants. Call write_comparison_report() to produce a full markdown
    summary comparing baseline vs. trained behaviour.
    """

    def __init__(self, log_dir: str = "logs/trajectories") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._current: Optional[EpisodeRecord] = None
        self._all_episodes: List[EpisodeRecord] = []

    # ── Episode lifecycle ──────────────────────────────────────────────────────

    def start_episode(
        self, task_id: str, model_label: str, seed: int = 0
    ) -> str:
        """
        Begin a new episode recording.

        Args:
            task_id:     One of the 5 CascadeGuard task IDs.
            model_label: Human-readable label, e.g. "random_baseline" or "grpo_trained".
            seed:        Episode seed used for reproducibility.

        Returns:
            Short episode ID string for reference.
        """
        ep_id = str(uuid.uuid4())[:8]
        self._current = EpisodeRecord(
            episode_id=ep_id,
            task_id=task_id,
            model_label=model_label,
            seed=seed,
            grader_score=0.0,
            started_at=datetime.utcnow().isoformat(),
        )
        return ep_id

    def record_step(
        self,
        obs,
        action_text: str,
        target: Optional[str],
        reward: float,
        metadata: Dict[str, Any],
        reasoning: str = "",
    ) -> None:
        """
        Record a single environment step.

        Args:
            obs:         CascadeObservation from env.step() or env.reset().
            action_text: The action_type string (e.g. "recover").
            target:      The target node_id or None.
            reward:      Step reward (obs.reward).
            metadata:    Info dict from obs.metadata (newly_failed, action_valid, etc.).
            reasoning:   Optional LLM <think> block text.
        """
        if self._current is None:
            raise RuntimeError("Call start_episode() before record_step().")

        hosp_nodes = [n for n in obs.nodes if n.sector == "hospital"]
        hosp_min = min((n.health for n in hosp_nodes), default=1.0)

        record = StepRecord(
            step=obs.step,
            action=action_text,
            target=target,
            reward=round(reward, 4),
            newly_failed=list(metadata.get("newly_failed", [])),
            completed_recoveries=list(metadata.get("completed_recoveries", [])),
            sector_health=dict(obs.sector_summary),
            hospital_min_health=round(hosp_min, 3),
            budget_remaining=round(obs.budget_remaining, 2),
            weather=obs.weather_forecast,
            action_valid=bool(metadata.get("action_valid", True)),
            reasoning=reasoning[:500],
        )
        self._current.steps.append(record)

    def end_episode(self, grader_score: float) -> EpisodeRecord:
        """
        Finalise the current episode and flush it to a JSON file.

        Args:
            grader_score: Final episode score from env.state.score.

        Returns:
            Completed EpisodeRecord.
        """
        if self._current is None:
            raise RuntimeError("No active episode. Call start_episode() first.")

        self._current.grader_score = round(grader_score, 4)
        self._current.ended_at = datetime.utcnow().isoformat()
        self._all_episodes.append(self._current)

        # Persist JSON artifact
        fname = (
            f"{self._current.task_id}_{self._current.model_label}"
            f"_{self._current.episode_id}.json"
        )
        out_path = self._log_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self._current.to_dict(), f, indent=2)

        ep = self._current
        self._current = None
        return ep

    # ── Reporting ─────────────────────────────────────────────────────────────

    def write_comparison_report(
        self, output_path: str = "logs/comparison.md"
    ) -> None:
        """
        Write a markdown comparison of all logged episodes grouped by task_id,
        then by model_label (baseline vs. trained).

        Produces a table per task/model that shows the step-by-step trajectory
        of the first representative episode plus aggregate statistics.
        """
        from collections import defaultdict

        by_task: Dict[str, List[EpisodeRecord]] = defaultdict(list)
        for ep in self._all_episodes:
            by_task[ep.task_id].append(ep)

        lines: List[str] = [
            "# CascadeGuard — Before vs After Training Comparison\n",
            f"Generated: {datetime.utcnow().isoformat()}\n",
        ]

        for task_id, episodes in sorted(by_task.items()):
            lines.append(f"\n## {task_id}\n")

            by_model: Dict[str, List[EpisodeRecord]] = defaultdict(list)
            for ep in episodes:
                by_model[ep.model_label].append(ep)

            for model_label, eps in sorted(by_model.items()):
                scores = [ep.grader_score for ep in eps]
                mean_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)

                lines.append(
                    f"### {model_label}  "
                    f"(n={len(eps)}, mean={mean_score:.3f}, "
                    f"min={min_score:.3f}, max={max_score:.3f})\n"
                )
                lines.append(
                    "| Step | Action | Target | Reward | Hosp Health"
                    " | Failures | Weather |"
                )
                lines.append("|---|---|---|---|---|---|---|")

                # Show first episode as the representative trajectory
                if eps:
                    for s in eps[0].steps:
                        failures = (
                            ", ".join(s.newly_failed) if s.newly_failed else "—"
                        )
                        lines.append(
                            f"| {s.step} | {s.action} | {s.target or '—'} "
                            f"| {s.reward:.3f} | {s.hospital_min_health:.2f} "
                            f"| {failures} | {s.weather} |"
                        )
                lines.append("")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[✓] Comparison report written to {out}")

    # ── Convenience ───────────────────────────────────────────────────────────

    def episode_count(self) -> int:
        return len(self._all_episodes)

    def score_summary(self) -> Dict[str, Dict[str, float]]:
        """Return {task_id: {model_label: mean_score}} for quick comparison."""
        from collections import defaultdict
        by_task: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for ep in self._all_episodes:
            by_task[ep.task_id][ep.model_label].append(ep.grader_score)
        return {
            task: {
                model: round(sum(scores) / len(scores), 4)
                for model, scores in models.items()
            }
            for task, models in by_task.items()
        }
