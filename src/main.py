"""Phone Agent CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger

from src.agent.agent import PhoneAgent
from src.config.settings import AgentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phone Agent — LLM-driven mobile assistant")
    parser.add_argument("task", nargs="*", help="Task to execute (e.g. 'Open WeChat')")
    parser.add_argument("--config", "-c", type=str, help="Path to YAML config file")
    parser.add_argument("--serial", "-s", type=str, help="Android device serial")
    parser.add_argument("--max-steps", type=int, help="Maximum steps per task")
    parser.add_argument("--no-safety", action="store_true", help="Disable safety layer")
    parser.add_argument("--no-fusion", action="store_true", help="Disable fusion framework")
    parser.add_argument("--no-router", action="store_true", help="Disable entropy router")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


async def main(task: str, config: AgentConfig) -> None:
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
    args = parse_args()

    if not args.task:
        print("Usage: python -m src.main 'your task here'")
        sys.exit(1)

    if not args.debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # Build config
    if args.config:
        config = AgentConfig.from_yaml(args.config)
    else:
        config = AgentConfig()

    if args.serial:
        config.device.serial = args.serial
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.no_safety:
        config.safety_enabled = False
    if args.no_fusion:
        config.fusion_enabled = False
    if args.no_router:
        config.router_enabled = False

    task = " ".join(args.task)
    asyncio.run(main(task, config))
