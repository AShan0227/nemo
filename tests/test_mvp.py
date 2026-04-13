"""Tests for MVP scenario definitions and evaluation."""

from __future__ import annotations

from types import SimpleNamespace

from src.agent.mvp import (
    evaluate_mvp_result,
    get_mvp_scenario,
    list_mvp_scenario_ids,
    summarize_mvp_reports,
)


def test_get_mvp_scenario_known():
    scenario = get_mvp_scenario("wechat_message")
    assert scenario.id == "wechat_message"
    assert "微信" in scenario.task


def test_get_mvp_scenario_unknown():
    try:
        get_mvp_scenario("unknown_case")
    except ValueError as exc:
        assert "Unknown scenario" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown scenario")


def test_evaluate_mvp_result_checkpoint_matching():
    scenario = get_mvp_scenario("taobao_search")
    steps = [
        SimpleNamespace(action="launch", reasoning="open taobao", params={}),
        SimpleNamespace(action="tap", reasoning="focus search input", params={}),
        SimpleNamespace(
            action="type_text",
            reasoning="输入 iPhone 手机壳",
            params={"text": "iPhone 手机壳"},
        ),
    ]
    result = SimpleNamespace(
        steps=steps,
        total_steps=3,
        status=SimpleNamespace(value="completed"),
    )
    report = evaluate_mvp_result(scenario, result)
    assert report["passed"] is True
    assert report["missing_checkpoints"] == []


def test_evaluate_mvp_result_reports_missing_checkpoints():
    scenario = get_mvp_scenario("settings_wifi")
    steps = [SimpleNamespace(action="tap", reasoning="open settings", params={})]
    result = SimpleNamespace(
        steps=steps,
        total_steps=1,
        status=SimpleNamespace(value="failed"),
    )
    report = evaluate_mvp_result(scenario, result)
    assert report["passed"] is False
    assert "home" in report["missing_checkpoints"]


def test_list_mvp_scenario_ids_sorted():
    ids = list_mvp_scenario_ids()
    assert ids == sorted(ids)
    assert "wechat_message" in ids


def test_summarize_mvp_reports():
    summary = summarize_mvp_reports(
        [
            {"passed": True},
            {"passed": False},
            {"passed": True},
        ]
    )
    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1
