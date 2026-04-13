"""Tests for action replay runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.device.actions import Action, ActionRecorder
from src.device.replay import ActionReplayer


@pytest.mark.asyncio
async def test_replay_respects_timing_with_speed_multiplier():
    recorder = ActionRecorder()
    recorder.record(timestamp_ms=1000, action=Action.tap(1, 2), success=True)
    recorder.record(timestamp_ms=1500, action=Action.tap(3, 4), success=True)
    recorder.record(timestamp_ms=2500, action=Action.tap(5, 6), success=True)

    executed: list[str] = []
    sleeps: list[float] = []

    async def fake_exec(action: Action) -> None:
        executed.append(action.type.value)

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    replayer = ActionReplayer(fake_exec, sleep_fn=fake_sleep)
    report = await replayer.replay_records(recorder.records, speed=2.0, preserve_timing=True)

    assert report.failed == 0
    assert report.succeeded == 3
    assert executed == ["tap", "tap", "tap"]
    assert sleeps == [0.25, 0.5]


@pytest.mark.asyncio
async def test_replay_collects_failures_without_stopping():
    recorder = ActionRecorder()
    recorder.record(timestamp_ms=1000, action=Action.tap(1, 2), success=True)
    recorder.record(timestamp_ms=1001, action=Action.tap(3, 4), success=True)
    recorder.record(timestamp_ms=1002, action=Action.tap(5, 6), success=True)

    index = {"value": 0}

    async def flaky_exec(action: Action) -> None:
        current = index["value"]
        index["value"] += 1
        if current == 1:
            raise RuntimeError("boom")

    replayer = ActionReplayer(flaky_exec)
    report = await replayer.replay_records(
        recorder.records,
        preserve_timing=False,
        stop_on_error=False,
    )

    assert report.succeeded == 2
    assert report.failed == 1
    assert report.failures[0].index == 1


@pytest.mark.asyncio
async def test_replay_file_uses_recorder_load(tmp_path: Path):
    recorder = ActionRecorder()
    recorder.record(timestamp_ms=1, action=Action.home(), success=True)
    path = recorder.save(tmp_path / "actions.json")
    seen: list[str] = []

    async def fake_exec(action: Action) -> None:
        seen.append(action.type.value)

    replayer = ActionReplayer(fake_exec)
    report = await replayer.replay_file(path, preserve_timing=False)

    assert report.total_actions == 1
    assert report.succeeded == 1
    assert seen == ["home"]
