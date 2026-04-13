"""Phone Agent CLI entry point."""

from __future__ import annotations

import asyncio
import sys

from loguru import logger

from src.agent.agent import PhoneAgent
from src.agent.mvp import evaluate_mvp_result, get_mvp_scenario
from src.config.settings import AgentConfig


async def main(task: str, scenario_id: str | None = None) -> None:
    config = AgentConfig()
    agent = PhoneAgent(config)
    scenario = get_mvp_scenario(scenario_id) if scenario_id else None
    task_to_run = scenario.task if scenario else task

    try:
        await agent.connect()
        result = await agent.execute(task_to_run)
        logger.info(f"Result: {result.status.value} in {result.total_steps} steps")
        logger.info(f"Summary: {result.summary}")
        for step in result.steps:
            status = "OK" if step.success else "FAIL"
            logger.info(f"  [{status}] Step {step.step}: {step.action} — {step.reasoning}")

        if scenario:
            report = evaluate_mvp_result(scenario, result)
            logger.info(
                "MVP Scenario [{}]: passed={} missing={}",
                report["scenario_id"],
                report["passed"],
                report["missing_checkpoints"],
            )
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python -m src.main 'your task here'")
        print("   or: python -m src.main --scenario wechat_message")
        sys.exit(1)

    scenario_id: str | None = None
    if args[0] == "--scenario":
        if len(args) < 2:
            print("Usage: python -m src.main --scenario <scenario_id>")
            sys.exit(1)
        scenario_id = args[1]
        task = ""
    else:
        task = " ".join(args)

    asyncio.run(main(task, scenario_id=scenario_id))
