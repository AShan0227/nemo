"""Prompt templates and tool schemas for the phone agent."""

from __future__ import annotations

from typing import Any

BASELINE_SYSTEM_PROMPT = """\
You are an AI phone assistant that controls an Android device.
Pick EXACTLY ONE next action.

Rules:
1. Only use element indices visible on the current screen.
2. Prefer reversible low-risk actions when uncertain.
3. Use `type_text` with both `index` and `text`.
4. Use `done` only when the user task is truly finished.
"""

SYSTEM_PROMPT = """\
You are an AI phone assistant that controls an Android device to complete user tasks.

Decision protocol (reflection-before-action):
1. Reflect briefly on user intent, current progress, and risk.
2. Pick EXACTLY ONE next action that is safest and most useful now.
3. If uncertain, prefer evidence-gathering actions (`scroll`, `wait`, `back`) over risky actions.

Hard constraints:
1. ONLY reference element indices shown in the current screen.
2. NEVER perform payments, permission grants, destructive deletes,
   or irreversible operations without explicit confirmation.
3. For text input, use `type_text` with both `index` and `text`.
4. For completion, use `done` with a concise factual summary.

Output contract:
- Preferred: call one tool.
- Fallback text mode: output strict JSON matching this schema:
{
  "reasoning": "short rationale",
  "action": "tap|type_text|scroll|long_press|double_tap|pinch_zoom|launch|back|home|wait|done",
  "params": {"...": "..."}
}
- Return no markdown, no code fences, and no extra prose outside JSON in fallback mode.
"""

DECISION_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "Task: Send \"到了\" to Xiao Wang in WeChat.\n\n"
            "Current Screen:\n"
            "App: com.tencent.mm\n"
            "[0] EditText \"Message\"\n"
            "[1] Button \"Send\""
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"reasoning":"Need to type the requested message before sending.",'
            '"action":"type_text","params":{"index":0,"text":"到了"}}'
        ),
    },
    {
        "role": "user",
        "content": (
            "Task: Open Wi-Fi settings.\n\n"
            "Current Screen:\n"
            "App: com.android.settings\n"
            "[0] TextView \"Network & Internet\"\n"
            "[1] TextView \"Apps\""
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"reasoning":"Network settings are the direct path to Wi-Fi.",'
            '"action":"tap","params":{"index":0}}'
        ),
    },
    {
        "role": "user",
        "content": (
            "Task: Find the refund option.\n\n"
            "Current Screen:\n"
            "App: com.shopping.app\n"
            "[0] TextView \"Order details\"\n"
            "[1] TextView \"Shipping\""
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"reasoning":"Refund is not visible yet; safest next step is to reveal more options.",'
            '"action":"scroll","params":{"direction":"down","amount":1}}'
        ),
    },
]


ACTION_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "tap",
            "description": "Tap one visible interactive element by index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Zero-based index from current screen interactive list.",
                    }
                },
                "required": ["index"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Focus input element by index and type exact text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Input field index on current screen.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Exact text to input.",
                    },
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
            "description": "Scroll current screen to discover more content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Scroll direction.",
                    },
                    "amount": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 1,
                        "description": "Relative scroll intensity (1=normal).",
                    },
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
            "description": "Long-press one visible interactive element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Element index to long-press.",
                    },
                    "duration_ms": {
                        "type": "integer",
                        "minimum": 200,
                        "default": 800,
                        "description": "Press duration in milliseconds.",
                    },
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
            "description": "Double-tap one visible interactive element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Element index to double-tap.",
                    },
                    "interval_ms": {
                        "type": "integer",
                        "minimum": 20,
                        "default": 120,
                        "description": "Gap between taps in milliseconds.",
                    },
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
            "description": "Pinch or zoom around an element index or coordinates.",
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
            "description": "Launch an app by Android package name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "package": {
                        "type": "string",
                        "description": "Android package name, e.g. com.tencent.mm.",
                    }
                },
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
            "description": "Wait briefly to let UI settle before next action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ms": {
                        "type": "integer",
                        "minimum": 100,
                        "default": 1000,
                        "description": "Wait duration in milliseconds.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Finish task when objective is achieved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Short factual completion summary.",
                    }
                },
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


def resolve_system_prompt(prompt_version: str | None) -> str:
    if prompt_version == "baseline_v1":
        return BASELINE_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def _format_history_block(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return ""

    lines = ["Recent trajectory (oldest -> newest):"]
    for item in history:
        step = item.get("step", "?")
        action = item.get("action", "")
        success = "ok" if item.get("success", False) else "fail"
        error = str(item.get("error", "")).strip()
        result_line = f"step={step}, action={action}, result={success}"
        if error:
            result_line = f"{result_line}, error={error[:120]}"
        lines.append(f"- {result_line}")
    return "\n".join(lines)


def build_decision_messages(
    system_prompt: str,
    task: str,
    screen_context: str,
    *,
    history: list[dict[str, Any]] | None = None,
    prompt_version: str | None = None,
) -> list[dict[str, str]]:
    """Build decision messages with optional few-shot and multi-step history."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # baseline prompt for A/B disables few-shot examples
    if prompt_version != "baseline_v1":
        messages.extend(DECISION_FEW_SHOT_EXAMPLES)

    history_block = _format_history_block(history)
    user_content = f"Task: {task}\n\nCurrent Screen:\n{screen_context}"
    if history_block:
        user_content = f"{user_content}\n\n{history_block}"
    user_content = f"{user_content}\n\nReturn exactly one next action."
    messages.append({"role": "user", "content": user_content})
    return messages
