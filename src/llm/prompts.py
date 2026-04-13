"""Prompt templates for the phone agent."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
You are an AI phone assistant that controls an Android device to complete user tasks.

You observe the current screen state and must pick EXACTLY ONE next action.

First think briefly about intent + safest progress.
Then call one action tool that best advances the task.

Important rules:
1. ONLY reference element indices shown in the current screen.
2. Prefer reversible, low-risk actions when uncertain.
3. For text input, choose `type_text` with both index and text.
4. For completion, call `done` with a concise summary.
5. NEVER perform financial transactions, permission grants,
   or destructive actions without confirmation.
6. If unsure, gather more evidence (e.g., scroll or wait) instead of risky actions.

You may optionally include short assistant text reasoning, but the tool call is authoritative.
"""


ACTION_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "tap",
            "description": "Tap an interactive element by index.",
            "parameters": {
                "type": "object",
                "properties": {"index": {"type": "integer", "minimum": 0}},
                "required": ["index"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Focus input element by index and type text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "text": {"type": "string"},
                },
                "required": ["index", "text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the current screen up or down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"]},
                    "amount": {"type": "integer", "minimum": 1, "default": 1},
                },
                "required": ["direction"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "long_press",
            "description": "Long press an interactive element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "duration_ms": {"type": "integer", "minimum": 200, "default": 800},
                },
                "required": ["index"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "double_tap",
            "description": "Double tap an interactive element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "interval_ms": {"type": "integer", "minimum": 20, "default": 120},
                },
                "required": ["index"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pinch_zoom",
            "description": "Pinch or zoom at element index or absolute coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "x": {"type": "integer", "minimum": 0},
                    "y": {"type": "integer", "minimum": 0},
                    "zoom_in": {"type": "boolean", "default": True},
                    "distance": {"type": "integer", "minimum": 20, "default": 220},
                    "duration_ms": {"type": "integer", "minimum": 80, "default": 350},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "launch",
            "description": "Launch app by Android package name.",
            "parameters": {
                "type": "object",
                "properties": {"package": {"type": "string"}},
                "required": ["package"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "back",
            "description": "Press Android back key.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "home",
            "description": "Press Android home key.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for a short duration to let UI settle.",
            "parameters": {
                "type": "object",
                "properties": {"ms": {"type": "integer", "minimum": 100, "default": 1000}},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Mark task as completed with a short summary.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    },
]

PLANNING_PROMPT = """\
Given the user's task and current screen, create a step-by-step plan.

Task: {task}
Current App: {current_app}
Current Screen: {screen_summary}

Provide a plan as a JSON array of steps:
[
  {{
    "step": 1,
    "action_name": "tap|type_text|scroll|back|home|launch|wait|done",
    "params": {{"index": 0}},
    "action": "short description",
    "expected_result": "what should happen"
  }},
  ...
]
"""
