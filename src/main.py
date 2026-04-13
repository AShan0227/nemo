"""Phone Agent CLI entry point."""

from __future__ import annotations

import asyncio
import sys

from loguru import logger

from src.agent.agent import PhoneAgent
from src.config.settings import AgentConfig


async def main(task: str) -> None:
    config = AgentConfig()
    agent = PhoneAgent(config)

    try:
        await agent.connect()
        result = await agent.execute(task)
        logger.info(f"Result: {result.status.value} in {result.total_steps} steps")
        logger.info(f"Summary: {result.summary}")
        for step in result.steps:
            status = "OK" if step.success else "FAIL"
            logger.info(f"  [{status}] Step {step.step}: {step.action} — {step.reasoning}")
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.main 'your task here'")
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    asyncio.run(main(task))
