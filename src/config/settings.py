"""Global configuration for Phone Agent."""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LLMProviderConfig(BaseModel):
    """Single provider endpoint/model config."""

    model: str
    base_url: str
    api_key: str = ""


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "qwen"
    model: str = "qwen-plus"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    providers: dict[str, LLMProviderConfig] = Field(
        default_factory=lambda: {
            "qwen": LLMProviderConfig(
                model="qwen-plus",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            "gpt": LLMProviderConfig(
                model="gpt-4o-mini",
                base_url="https://api.openai.com/v1",
            ),
            "claude": LLMProviderConfig(
                model="claude-3-7-sonnet-latest",
                base_url="https://api.anthropic.com/v1",
            ),
            "deepseek": LLMProviderConfig(
                model="deepseek-chat",
                base_url="https://api.deepseek.com/v1",
            ),
        }
    )
    fallback_providers: list[str] = Field(default_factory=list)
    temperature: float = 0.3
    max_tokens: int = 4096
    request_timeout_s: float = 45.0
    retry_count: int = 2
    retry_backoff_s: float = 1.0
    enable_stream: bool = False
    tool_use_enabled: bool = True

    @field_validator("api_key", mode="before")
    @classmethod
    def _load_api_key(cls, v: str) -> str:
        return v or os.environ.get("LLM_API_KEY", "")


class DeviceConfig(BaseModel):
    """Android device connection configuration."""

    serial: Optional[str] = None  # noqa: UP045 (py39 compatibility)
    adb_host: str = "127.0.0.1"
    adb_port: int = 5037
    screenshot_dir: str = "./screenshots"
    screen_width: Optional[int] = None  # noqa: UP045 (py39 compatibility)
    screen_height: Optional[int] = None  # noqa: UP045 (py39 compatibility)


class AgentConfig(BaseModel):
    """Top-level agent configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    max_steps: int = 30
    action_delay_ms: int = 500
    step_retry_count: int = 1
    verify_enabled: bool = True
    verify_diff_threshold: float = 0.01
    entropy_threshold_low: float = 0.3
    entropy_threshold_high: float = 0.7
    safety_enabled: bool = True
    safety_rules_path: Optional[str] = None  # noqa: UP045 (py39 compatibility)
    safety_audit_log_path: Optional[str] = "./logs/safety_audit.log"  # noqa: UP045
    graph_persist_path: Optional[str] = "./data/screen_graph.json"  # noqa: UP045
    fusion_enabled: bool = True
    router_enabled: bool = True
    planner_enabled: bool = True
    planner_confidence_threshold: float = 0.7
    ocr_enabled: bool = False
    ocr_min_confidence: float = 0.45
    fusion_top_k: int = 5

    @classmethod
    def from_yaml(cls, path: str) -> AgentConfig:
        """Load config from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
