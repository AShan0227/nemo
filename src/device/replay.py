"""Action replay runtime for recorded execution traces."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.device.actions import Action, ActionRecord, ActionRecorder


@dataclass
class ReplayFailure:
    """Single replay failure detail."""

    index: int
    action_type: str
    error: str


@dataclass
class ReplayReport:
    """Replay summary for logging and diagnostics."""

    total_actions: int
    succeeded: int
    failed: int
    failures: list[ReplayFailure] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total_actions if self.total_actions > 0 else 0.0


class ActionReplayer:
    """Replay recorded actions with optional timing preservation."""

    def __init__(
        self,
        executor: Callable[[Action], Awaitable[None]],
        *,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._executor = executor
        self._sleep = sleep_fn

    async def replay_records(
        self,
        records: list[ActionRecord],
        *,
        speed: float = 1.0,
        preserve_timing: bool = True,
        stop_on_error: bool = True,
    ) -> ReplayReport:
        if speed <= 0:
            raise ValueError("speed must be > 0")

        failures: list[ReplayFailure] = []
        succeeded = 0
        previous_timestamp: int | None = None

        for index, record in enumerate(records):
            if preserve_timing and previous_timestamp is not None:
                gap_ms = max(0, record.timestamp_ms - previous_timestamp)
                if gap_ms > 0:
                    await self._sleep((gap_ms / 1000) / speed)

            previous_timestamp = record.timestamp_ms

            try:
                await self._executor(record.action)
                succeeded += 1
            except Exception as exc:
                failures.append(
                    ReplayFailure(
                        index=index,
                        action_type=record.action.type.value,
                        error=str(exc),
                    )
                )
                if stop_on_error:
                    break

        total = len(records)
        failed = len(failures)
        return ReplayReport(
            total_actions=total,
            succeeded=succeeded,
            failed=failed,
            failures=failures,
        )

    async def replay_file(
        self,
        path: str | Path,
        *,
        speed: float = 1.0,
        preserve_timing: bool = True,
        stop_on_error: bool = True,
    ) -> ReplayReport:
        recorder = ActionRecorder.load(path)
        return await self.replay_records(
            recorder.records,
            speed=speed,
            preserve_timing=preserve_timing,
            stop_on_error=stop_on_error,
        )
