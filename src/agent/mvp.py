"""MVP single-app scenarios and lightweight completion checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MVPScenario:
    """Acceptance scenario used in milestone #3."""

    id: str
    title: str
    task: str
    checkpoints: tuple[str, ...]


MVP_SCENARIOS: dict[str, MVPScenario] = {
    "wechat_message": MVPScenario(
        id="wechat_message",
        title="WeChat message",
        task="给张三发一条微信说明天开会",
        checkpoints=("launch", "search", "type_text", "send"),
    ),
    "settings_wifi": MVPScenario(
        id="settings_wifi",
        title="Settings WiFi",
        task="打开 WiFi 设置",
        checkpoints=("home", "settings", "wifi"),
    ),
    "taobao_search": MVPScenario(
        id="taobao_search",
        title="Taobao search",
        task="在淘宝搜索 iPhone 手机壳",
        checkpoints=("launch", "taobao", "search", "type_text"),
    ),
}


def list_mvp_scenario_ids() -> list[str]:
    return sorted(MVP_SCENARIOS.keys())


def get_mvp_scenario(scenario_id: str) -> MVPScenario:
    key = scenario_id.strip().lower()
    if key not in MVP_SCENARIOS:
        available = ", ".join(sorted(MVP_SCENARIOS))
        raise ValueError(f"Unknown scenario `{scenario_id}`. Available: {available}")
    return MVP_SCENARIOS[key]


def evaluate_mvp_result(
    scenario: MVPScenario,
    result: Any,
) -> dict[str, Any]:
    """Evaluate whether scenario checkpoints appear in execution trace."""
    steps = getattr(result, "steps", [])
    traces: list[str] = []
    for step in steps:
        action = str(getattr(step, "action", "")).lower()
        reasoning = str(getattr(step, "reasoning", "")).lower()
        params = str(getattr(step, "params", "")).lower()
        traces.append(f"{action} {reasoning} {params}")

    missing: list[str] = []
    for checkpoint in scenario.checkpoints:
        needle = checkpoint.lower()
        if not any(needle in trace for trace in traces):
            missing.append(checkpoint)

    status_obj = getattr(result, "status", "")
    status_value = getattr(status_obj, "value", status_obj)

    return {
        "scenario_id": scenario.id,
        "title": scenario.title,
        "passed": not missing,
        "missing_checkpoints": missing,
        "total_steps": int(getattr(result, "total_steps", 0)),
        "status": str(status_value),
    }


def summarize_mvp_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(reports)
    passed = sum(1 for report in reports if bool(report.get("passed")))
    failed = total - passed
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed / total) if total > 0 else 0.0,
    }
