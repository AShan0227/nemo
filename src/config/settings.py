"""Global configuration for Phone Agent."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    model: str = "qwen-plus"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096

    @field_validator("api_key", mode="before")
    @classmethod
    def _load_api_key(cls, v: str) -> str:
        return v or os.environ.get("LLM_API_KEY", "")


class DeviceConfig(BaseModel):
    """Android device connection configuration."""

    serial: Optional[str] = None  # auto-detect if None
    adb_host: str = "127.0.0.1"
    adb_port: int = 5037
    screenshot_dir: str = "./screenshots"
    screen_width: Optional[int] = None   # override auto-detect
    screen_height: Optional[int] = None  # override auto-detect


class AgentConfig(BaseModel):
    """Top-level agent configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    max_steps: int = 30
    action_delay_ms: int = 500
    entropy_threshold_low: float = 0.3
    entropy_threshold_high: float = 0.7
    safety_enabled: bool = True
    fusion_enabled: bool = True
    router_enabled: bool = True
    planner_enabled: bool = True
    planner_confidence_threshold: float = 0.7

    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        """Load config from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
