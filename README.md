# Phone Agent

LLM-driven AI phone assistant with autonomous screen understanding and task execution.

Inspired by Doubao Phone Assistant and Siri — control your Android phone with natural language.

## Architecture

```
User Command ("Send a message to John")
        │
        ▼
┌─────────────────────────────────┐
│          PhoneAgent             │
│  ┌───────────┐ ┌────────────┐  │
│  │  Planner  │ │  Entropy   │  │
│  │  (A*/ACO) │ │  Router    │  │
│  └─────┬─────┘ └─────┬──────┘  │
│        └──────┬───────┘         │
│               ▼                 │
│        ┌────────────┐           │
│        │  LLM Brain │           │
│        └─────┬──────┘           │
│              ▼                  │
│        ┌────────────┐           │
│        │Safety Layer│           │
│        └─────┬──────┘           │
│              ▼                  │
│  ┌───────────────────────────┐  │
│  │    Screen Understanding   │  │
│  │ UI Hierarchy + OCR + DS   │  │
│  └───────────┬───────────────┘  │
│              ▼                  │
│  ┌───────────────────────────┐  │
│  │    Device Controller      │  │
│  │    (ADB / Accessibility)  │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
        │
        ▼
   Android Device
```

## Modules

| Module | Path | Description |
|--------|------|-------------|
| Agent Core | `src/agent/` | Main agent loop, task planner |
| Screen Understanding | `src/screen/` | UI hierarchy parser, DS fusion |
| Device Control | `src/device/` | ADB controller, action primitives |
| LLM Integration | `src/llm/` | LLM client, entropy router, prompts |
| Safety | `src/safety/` | Invariant constraints (financial, permission, deletion) |
| Knowledge Graph | `src/knowledge/` | Screen state graph with A* and ACO |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Connect your Android device via USB, ensure ADB is enabled
adb devices

# Run
python -m src.main "Open WeChat and send hello to John"

# Run milestone MVP scenario (with checkpoint report)
python -m src.main --scenario wechat_message

# Save executed actions, then replay
python -m src.main "打开设置" --record-out ./artifacts/actions.json
python -m src.main --replay ./artifacts/actions.json --replay-speed 2.0

# Run all built-in MVP scenarios and export summary report
python -m src.main --suite --suite-report ./artifacts/mvp_suite_report.json

# Test
pytest tests/
```

## Configuration

Set your LLM API key:
```bash
export LLM_API_KEY="your-key"
```

Edit `src/config/settings.py` for model, device, and safety settings.

Safety YAML example (optional):
```yaml
# configs/safety_rules.example.yaml
rules:
  financial_guard:
    enabled: true
  message_send_guard:
    enabled: true
audit_log_path: "./logs/safety_audit.log"
```

Then set:
```python
AgentConfig(
    llm=LLMConfig(
        request_timeout_s=45,
        retry_count=2,
        retry_backoff_s=1.0,
        enable_stream=False,
    ),
    safety_enabled=True,
    safety_rules_path="configs/safety_rules.example.yaml",
    step_retry_count=1,
    verify_enabled=True,
    verify_diff_threshold=0.01,
    ocr_enabled=True,
    ocr_min_confidence=0.45,
    fusion_top_k=5,
)
```

## Research

See `docs/` and `research/` for deep technical research on physics-inspired and biology-inspired mechanisms behind this agent.

## License

MIT
