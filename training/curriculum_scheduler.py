"""
curriculum_scheduler.py — Adaptive curriculum for CascadeGuard training
=========================================================================
Tracks per-task rolling mean rewards and graduates the training task
when the agent demonstrates consistent mastery above a threshold.

Usage:
    scheduler = CurriculumScheduler()
    scheduler.record(score=0.78)
    next_task = scheduler.current_task    # may have advanced to next stage
    print(scheduler.summary())
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List


# ── Curriculum definition ──────────────────────────────────────────────────────
# (task_id, graduate_threshold, min_episodes_before_graduation)
CURRICULUM_STAGES: List[tuple] = [
    ("task_easy",         0.75, 10),
    ("task_gen_blackout", 0.65, 10),
    ("task_medium",       0.55, 15),
    ("task_hard",         0.50, 20),
    ("task_cyberattack",  0.45, 20),
]


@dataclass
class StageStats:
    task_id: str
    threshold: float
    min_episodes: int
    scores: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    graduated: bool = False
    episodes_seen: int = 0

    @property
    def rolling_mean(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    @property
    def can_graduate(self) -> bool:
        if self.graduated:
            return False
        if self.episodes_seen < self.min_episodes:
            return False
        return self.rolling_mean >= self.threshold


class CurriculumScheduler:
    """
    Manages adaptive curriculum progression for CascadeGuard training.

    Records episode scores for the current stage and automatically
    advances to the next stage when the rolling mean exceeds the
    graduation threshold.
    """

    def __init__(self) -> None:
        self._stages: List[StageStats] = [
            StageStats(task_id=t, threshold=th, min_episodes=me)
            for t, th, me in CURRICULUM_STAGES
        ]
        self._stage_idx: int = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_task(self) -> str:
        """Task ID of the currently active curriculum stage."""
        return self._stages[self._stage_idx].task_id

    @property
    def stage_index(self) -> int:
        """0-indexed current stage (0 = easiest, 4 = hardest)."""
        return self._stage_idx

    @property
    def is_final_stage(self) -> bool:
        """True when the agent is on the last (hardest) task."""
        return self._stage_idx == len(self._stages) - 1

    @property
    def current_threshold(self) -> float:
        """Graduation threshold for the current stage."""
        return self._stages[self._stage_idx].threshold

    @property
    def current_rolling_mean(self) -> float:
        """Rolling mean score for the current stage (window=20)."""
        return self._stages[self._stage_idx].rolling_mean

    # ── Core methods ──────────────────────────────────────────────────────────

    def record(self, score: float) -> bool:
        """
        Record an episode score for the current stage.

        Args:
            score: Grader score in (0, 1) from env.state.score.

        Returns:
            True if the stage just graduated (task will change next call).
        """
        stage = self._stages[self._stage_idx]
        stage.scores.append(score)
        stage.episodes_seen += 1

        if stage.can_graduate and not self.is_final_stage:
            stage.graduated = True
            self._stage_idx += 1
            print(
                f"\n[Curriculum] ✓ Graduated '{stage.task_id}' "
                f"(rolling_mean={stage.rolling_mean:.3f} ≥ {stage.threshold})"
            )
            print(f"[Curriculum] → Now training on: '{self.current_task}'\n")
            return True
        return False

    def get_seeds(self, split: str = "train") -> List[int]:
        """Return training/validation seeds for the current task."""
        from cascade_guard.tasks import TASK_SEED_SPLITS
        return TASK_SEED_SPLITS[self.current_task][split]

    def summary(self) -> str:
        """Human-readable curriculum progress table."""
        lines = ["=== Curriculum Summary ==="]
        for i, stage in enumerate(self._stages):
            if i == self._stage_idx:
                status = "CURRENT"
            elif stage.graduated:
                status = "DONE   "
            else:
                status = "pending"
            lines.append(
                f"  [{status}] {stage.task_id:<22} "
                f"mean={stage.rolling_mean:.3f}  "
                f"eps={stage.episodes_seen:>3}  "
                f"thresh={stage.threshold}"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset scheduler to stage 0 (for re-training from scratch)."""
        self._stage_idx = 0
        for stage in self._stages:
            stage.scores.clear()
            stage.graduated = False
            stage.episodes_seen = 0
