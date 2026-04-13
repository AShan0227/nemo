"""Global configuration for Phone Agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class DeviceConfig(BaseModel):
    """Android device connection configuration."""

    serial: str | None = None  # auto-detect if None
    adb_host: str = "127.0.0.1"
    adb_port: int = 5037
    screenshot_dir: str = "./screenshots"


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
    safety_rules_path: str | None = None
    safety_audit_log_path: str | None = "./logs/safety_audit.log"
    ocr_enabled: bool = False
    ocr_min_confidence: float = 0.45
    fusion_top_k: int = 5
