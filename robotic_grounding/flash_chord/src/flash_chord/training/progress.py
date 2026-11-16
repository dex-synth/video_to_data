# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact terminal progress for local training runs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol


class TrainingProgress(Protocol):
    """Progress reporter driven by completed environment steps."""

    def start(self, completed_environment_steps: int) -> None:
        """Start reporting from a new or resumed training state."""
        ...

    def update(self, completed_environment_steps: int) -> None:
        """Report a later training state when its display boundary is crossed."""
        ...


class ConsoleTrainingProgress:
    """Print bounded progress at one-percent environment-step intervals."""

    def __init__(self, total_environment_steps: int, checkpoint_dir: str | Path) -> None:
        if total_environment_steps <= 0:
            raise ValueError(f"total_environment_steps must be positive, got {total_environment_steps}")
        self.total_environment_steps = total_environment_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self._start_environment_steps = 0
        self._last_environment_steps = -1
        self._next_percent = 0
        self._start_time: float | None = None
        self.compile_seconds: float | None = None

    def start(self, completed_environment_steps: int) -> None:
        """Print the initial state and initialize throughput/ETA timing."""
        self._validate_steps(completed_environment_steps)
        self._start_environment_steps = completed_environment_steps
        self._last_environment_steps = completed_environment_steps
        self._next_percent = min(100, completed_environment_steps * 100 // self.total_environment_steps + 1)
        self._start_time = time.perf_counter()
        print(
            f"training progress: {completed_environment_steps:,}/{self.total_environment_steps:,} "
            f"environment steps ({self._percent(completed_environment_steps):.1f}%); "
            f"checkpoints: {self.checkpoint_dir}",
            flush=True,
        )

    def update(
        self,
        completed_environment_steps: int,
        *,
        invalid_episodes=None,
        completed_episodes=None,
    ) -> None:
        """Print after crossing the next whole percent, and always at completion."""
        if self._start_time is None:
            raise RuntimeError("training progress must be started before it is updated")
        self._validate_steps(completed_environment_steps)
        if completed_environment_steps < self._last_environment_steps:
            raise ValueError(
                "completed_environment_steps must not decrease: "
                f"{completed_environment_steps} < {self._last_environment_steps}"
            )
        percent = self._percent(completed_environment_steps)
        terminal = completed_environment_steps == self.total_environment_steps
        if not terminal and percent < self._next_percent:
            return
        if completed_environment_steps == self._last_environment_steps:
            return

        elapsed_seconds = time.perf_counter() - self._start_time
        completed_since_start = completed_environment_steps - self._start_environment_steps
        rate = completed_since_start / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        remaining = self.total_environment_steps - completed_environment_steps
        eta = _format_duration(remaining / rate) if rate > 0.0 else "calculating"
        episode_summary = ""
        if invalid_episodes is not None and completed_episodes is not None:
            failed, completed = int(invalid_episodes), int(completed_episodes)
            fraction = f"{100 * failed / completed:.2f}%" if completed else "n/a"
            episode_summary = f"; invalid episodes: {failed:,}/{completed:,} ({fraction})"
        print(
            f"training progress: {completed_environment_steps:,}/{self.total_environment_steps:,} "
            f"environment steps ({percent:.1f}%); {rate:,.0f} env steps/s; "
            f"elapsed {_format_duration(elapsed_seconds)}; ETA {eta}{episode_summary}",
            flush=True,
        )
        self._last_environment_steps = completed_environment_steps
        self._next_percent = min(100, int(percent) + 1)

    def compiled(self, seconds: float) -> None:
        """Record the compilation of the training executables, which happens after start."""
        self.compile_seconds = float(seconds)

    def timing(self) -> dict[str, float | int | None]:
        """Wall time since start, its compilation share and the last displayed step count; None before start."""
        if self._start_time is None:
            return {"environment_steps_start": None, "last_reported_environment_steps": None,
                    "elapsed_s": None, "compile_s": None, "loop_s": None}
        elapsed = time.perf_counter() - self._start_time
        compile_seconds = self.compile_seconds
        return {"environment_steps_start": self._start_environment_steps,
                "last_reported_environment_steps": self._last_environment_steps,
                "elapsed_s": elapsed, "compile_s": compile_seconds,
                "loop_s": elapsed - compile_seconds if compile_seconds is not None else None}

    def _percent(self, completed_environment_steps: int) -> float:
        return 100.0 * completed_environment_steps / self.total_environment_steps

    def _validate_steps(self, completed_environment_steps: int) -> None:
        if not 0 <= completed_environment_steps <= self.total_environment_steps:
            raise ValueError(
                f"completed_environment_steps must be in [0, {self.total_environment_steps}], "
                f"got {completed_environment_steps}"
            )


def _format_duration(seconds: float) -> str:
    """Format a non-negative duration without subsecond terminal churn."""
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
