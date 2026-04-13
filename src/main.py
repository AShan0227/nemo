"""Phone Agent CLI entry point."""

from __future__ import annotations

import asyncio
from argparse import ArgumentParser

from loguru import logger

from src.agent.agent import PhoneAgent
from src.agent.mvp import evaluate_mvp_result, get_mvp_scenario
from src.config.settings import AgentConfig


async def main(
    task: str,
    *,
    scenario_id: str | None = None,
    replay_path: str | None = None,
    replay_speed: float = 1.0,
    record_out: str | None = None,
) -> None:
    config = AgentConfig()
    agent = PhoneAgent(config)
    scenario = get_mvp_scenario(scenario_id) if scenario_id else None
    task_to_run = scenario.task if scenario else task

    try:
        await agent.connect()
        if replay_path:
            report = await agent.replay_recording(replay_path, speed=replay_speed)
            logger.info(
                "Replay report: total={} success={} failed={} rate={:.2f}",
                report.total_actions,
                report.succeeded,
                report.failed,
                report.success_rate,
            )
            for failure in report.failures:
                logger.warning(
                    "  [FAIL] action #{} {}: {}",
                    failure.index,
                    failure.action_type,
                    failure.error,
                )
            return

        result = await agent.execute(task_to_run)
        logger.info(f"Result: {result.status.value} in {result.total_steps} steps")
        logger.info(f"Summary: {result.summary}")
        for step in result.steps:
            status = "OK" if step.success else "FAIL"
            logger.info(
                "  [{}] Step {}: {} (attempts={}, {:.1f}ms) — {}",
                status,
                step.step,
                step.action,
                step.attempts,
                step.duration_ms,
                step.reasoning,
            )

        if record_out:
            path = agent.action_recorder.save(record_out)
            logger.info("Saved action recording: {}", path)

        if scenario:
            mvp_report = evaluate_mvp_result(scenario, result)
            logger.info(
                "MVP Scenario [{}]: passed={} missing={}",
                mvp_report["scenario_id"],
                mvp_report["passed"],
                mvp_report["missing_checkpoints"],
            )
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise


if __name__ == "__main__":
    parser = ArgumentParser(description="Phone Agent CLI")
    parser.add_argument("task", nargs="*", help="Natural language task")
    parser.add_argument(
        "--scenario",
        dest="scenario_id",
        help="Run built-in MVP scenario (wechat_message|settings_wifi|taobao_search)",
    )
    parser.add_argument("--replay", dest="replay_path", help="Replay recording JSON file")
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier (e.g. 2.0 = 2x faster)",
    )
    parser.add_argument("--record-out", dest="record_out", help="Save executed actions to JSON")
    parsed = parser.parse_args()

    task_text = " ".join(parsed.task).strip()
    if not parsed.replay_path and not parsed.scenario_id and not task_text:
        parser.error("Provide a task, or use --scenario, or --replay")

    asyncio.run(
        main(
            task_text,
            scenario_id=parsed.scenario_id,
            replay_path=parsed.replay_path,
            replay_speed=parsed.replay_speed,
            record_out=parsed.record_out,
        )
    )
