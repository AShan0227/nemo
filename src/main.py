"""Phone Agent CLI entry point."""

from __future__ import annotations

import asyncio
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

from loguru import logger

from src.agent.agent import PhoneAgent
from src.agent.mvp import (
    evaluate_mvp_result,
    get_mvp_scenario,
    list_mvp_scenario_ids,
    summarize_mvp_reports,
)
from src.config.settings import AgentConfig


async def main(
    task: str,
    config: AgentConfig,
    *,
    scenario_id: str | None = None,
    replay_path: str | None = None,
    replay_speed: float = 1.0,
    record_out: str | None = None,
    suite: bool = False,
    suite_report_path: str | None = None,
) -> None:
    agent = PhoneAgent(config)
    scenario = get_mvp_scenario(scenario_id) if scenario_id else None
    task_to_run = scenario.task if scenario else task

    try:
        await agent.connect()

        if replay_path:
            report = await agent.replay_recording(replay_path, speed=replay_speed)
            logger.info(
                "Replay report: total={} success={} failed={} rate={:.2f}",
                report.total_actions, report.succeeded, report.failed, report.success_rate,
            )
            for failure in report.failures:
                logger.warning(
                    "  [FAIL] action #{} {}: {}",
                    failure.index, failure.action_type, failure.error,
                )
            return

        if suite:
            suite_reports: list[dict[str, object]] = []
            for sid in list_mvp_scenario_ids():
                scenario_obj = get_mvp_scenario(sid)
                logger.info("Running scenario [{}] {}", scenario_obj.id, scenario_obj.title)
                scenario_result = await agent.execute(scenario_obj.task)
                report = evaluate_mvp_result(scenario_obj, scenario_result)
                suite_reports.append(report)
                logger.info(
                    "  => passed={} status={} missing={}",
                    report["passed"], report["status"], report["missing_checkpoints"],
                )

            summary = summarize_mvp_reports(suite_reports)
            logger.info(
                "Suite summary: total={} passed={} failed={} pass_rate={:.2f}",
                summary["total"], summary["passed"], summary["failed"], summary["pass_rate"],
            )
            if suite_report_path:
                payload = {"summary": summary, "reports": suite_reports}
                target = Path(suite_report_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("Saved suite report: {}", target)
            return

        result = await agent.execute(task_to_run)
        logger.info(f"Result: {result.status.value} in {result.total_steps} steps")
        logger.info(f"Summary: {result.summary}")
        for step in result.steps:
            status = "OK" if step.success else "FAIL"
            logger.info(
                "  [{}] Step {}: {} (attempts={}, {:.1f}ms) — {}",
                status, step.step, step.action, step.attempts, step.duration_ms, step.reasoning,
            )

        if record_out:
            path = agent.action_recorder.save(record_out)
            logger.info("Saved action recording: {}", path)

        if scenario:
            mvp_report = evaluate_mvp_result(scenario, result)
            logger.info(
                "MVP Scenario [{}]: passed={} missing={}",
                mvp_report["scenario_id"], mvp_report["passed"], mvp_report["missing_checkpoints"],
            )
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise


if __name__ == "__main__":
    parser = ArgumentParser(description="Phone Agent — LLM-driven mobile assistant")
    parser.add_argument("task", nargs="*", help="Natural language task")
    parser.add_argument("--config", "-c", type=str, help="Path to YAML config file")
    parser.add_argument("--serial", "-s", type=str, help="Android device serial")
    parser.add_argument("--max-steps", type=int, help="Maximum steps per task")
    parser.add_argument("--no-safety", action="store_true", help="Disable safety layer")
    parser.add_argument("--no-fusion", action="store_true", help="Disable fusion framework")
    parser.add_argument("--no-router", action="store_true", help="Disable entropy router")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--scenario", dest="scenario_id",
                        help="Run built-in MVP scenario (wechat_message|settings_wifi|taobao_search)")
    parser.add_argument("--suite", action="store_true", help="Run all built-in MVP scenarios")
    parser.add_argument("--suite-report", dest="suite_report_path", help="Save MVP suite report JSON")
    parser.add_argument("--replay", dest="replay_path", help="Replay recording JSON file")
    parser.add_argument("--replay-speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--record-out", dest="record_out", help="Save executed actions to JSON")
    args = parser.parse_args()

    if not args.replay_path and not args.scenario_id and not args.suite and not args.task:
        parser.error("Provide a task, or use --scenario, --replay, or --suite")

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

    task_text = " ".join(args.task).strip() if args.task else ""
    asyncio.run(main(
        task_text, config,
        scenario_id=args.scenario_id,
        replay_path=args.replay_path,
        replay_speed=args.replay_speed,
        record_out=args.record_out,
        suite=args.suite,
        suite_report_path=args.suite_report_path,
    ))
