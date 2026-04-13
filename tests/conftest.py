"""Shared test fixtures — mock ADB, mock LLM, sample screens."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.config.settings import AgentConfig
from src.device.actions import Action, ActionType
from src.screen.parser import ScreenState, UIElement


# --- Sample XML screens ---

SETTINGS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="" class="android.widget.FrameLayout"
        content-desc="" clickable="false" enabled="true" bounds="[0,0][1080,1920]">
    <node index="0" text="Settings" resource-id="com.android.settings:id/title"
          class="android.widget.TextView" content-desc=""
          clickable="true" enabled="true" bounds="[0,0][540,100]" />
    <node index="1" text="Wi-Fi" resource-id="com.android.settings:id/wifi"
          class="android.widget.TextView" content-desc=""
          clickable="true" enabled="true" bounds="[0,100][540,200]" />
    <node index="2" text="Bluetooth" resource-id="com.android.settings:id/bt"
          class="android.widget.TextView" content-desc=""
          clickable="true" enabled="true" bounds="[0,200][540,300]" />
  </node>
</hierarchy>"""

WIFI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="" class="android.widget.FrameLayout"
        content-desc="" clickable="false" enabled="true" bounds="[0,0][1080,1920]">
    <node index="0" text="Wi-Fi" resource-id="com.android.settings:id/title"
          class="android.widget.TextView" content-desc=""
          clickable="false" enabled="true" bounds="[0,0][540,100]" />
    <node index="1" text="" resource-id="com.android.settings:id/switch_widget"
          class="android.widget.Switch" content-desc="Wi-Fi toggle"
          clickable="true" checked="true" enabled="true" bounds="[800,0][1080,100]" />
  </node>
</hierarchy>"""

WECHAT_CHAT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" class="android.widget.FrameLayout"
        clickable="false" enabled="true" bounds="[0,0][1080,1920]">
    <node index="0" text="John" class="android.widget.TextView"
          clickable="false" enabled="true" bounds="[0,0][540,100]" />
    <node index="1" text="" class="android.widget.EditText"
          content-desc="Type a message" clickable="true"
          enabled="true" bounds="[0,1700][900,1800]" />
    <node index="2" text="Send" class="android.widget.Button"
          clickable="true" enabled="true" bounds="[900,1700][1080,1800]" />
  </node>
</hierarchy>"""

PURCHASE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" class="android.widget.FrameLayout"
        clickable="false" enabled="true" bounds="[0,0][1080,1920]">
    <node index="0" text="Confirm purchase $99.99" class="android.widget.TextView"
          clickable="false" enabled="true" bounds="[0,800][1080,900]" />
    <node index="1" text="Pay Now" class="android.widget.Button"
          clickable="true" enabled="true" bounds="[300,1000][780,1100]" />
  </node>
</hierarchy>"""


def make_llm_response(action: str, params: dict | None = None, reasoning: str = "test") -> str:
    """Create a mock LLM JSON response string."""
    return json.dumps({
        "reasoning": reasoning,
        "action": action,
        "params": params or {},
    })


@pytest.fixture
def agent_config() -> AgentConfig:
    """Config with safety enabled, small max steps for testing."""
    return AgentConfig(
        max_steps=5,
        action_delay_ms=0,
        safety_enabled=True,
        fusion_enabled=False,  # disable fusion in unit tests for speed
    )
