"""Global configuration for Phone Agent."""

import os
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    model: str = "qwen-plus"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096
    request_timeout_s: float = 45.0
    retry_count: int = 2
    retry_backoff_s: float = 1.0
    enable_stream: bool = False

    @field_validator("api_key", mode="before")
    @classmethod
    def _load_api_key(cls, v: str) -> str:
        return v or os.environ.get("LLM_API_KEY", "")


class DeviceConfig(BaseModel):
    """Android device connection configuration."""

    serial: Optional[str] = None
    adb_host: str = "127.0.0.1"
    adb_port: int = 5037
    screenshot_dir: str = "./screenshots"
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None


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
    safety_rules_path: Optional[str] = None
    safety_audit_log_path: Optional[str] = "./logs/safety_audit.log"
    fusion_enabled: bool = True
    router_enabled: bool = True
    planner_enabled: bool = True
    planner_confidence_threshold: float = 0.7
    ocr_enabled: bool = False
    ocr_min_confidence: float = 0.45
    fusion_top_k: int = 5
    # Research mechanisms
    homeostasis_enabled: bool = True
    immune_enabled: bool = False  # needs training data first
    immune_anomaly_threshold: int = 3
    inertia_enabled: bool = True
    inertia_base: float = 0.3
    explorer_enabled: bool = False  # opt-in Boltzmann exploration
    explorer_temperature: float = 1.0
    phase_detector_enabled: bool = True
    circadian_enabled: bool = True
    graph_persist_path: Optional[str] = None  # e.g. "./data/graph.json"

    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        """Load config from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
