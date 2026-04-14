<p align="center">
  <h1 align="center">🧠 Nemo</h1>
  <p align="center"><b>Give your phone a nervous system.</b></p>
  <p align="center">
    The first open-source AI agent that runs <i>entirely on your phone</i>.<br>
    On-device Gemma model. Zero cloud. Zero API fees. Your data never leaves your device.
  </p>
  <p align="center">
    <a href="https://github.com/AShan0227/phone-agent/releases">📱 Download APK</a> ·
    <a href="#nerve">🧬 What is NERVE?</a> ·
    <a href="#quickstart">⚡ Quick Start</a> ·
    <a href="#demo">🎬 Demo</a>
  </p>
</p>

---

> **Siri made you wait 13 years. Google Gemini still runs in the cloud.**
> Nemo is an AI agent that lives *inside* your phone — open-source, free forever, works offline.

## What can it do?

Tell Nemo what you want in natural language. It reads your screen, thinks, and acts — just like you would.

```
"打开微信给张三发消息说明天开会"     → Opens WeChat, finds contact, types message, sends
"Turn off WiFi"                    → Opens Settings, finds WiFi toggle, taps it
"在淘宝搜索 iPhone 手机壳"          → Opens Taobao, taps search, types query, searches
```

<a id="nerve"></a>
## 🧬 NERVE — What makes Nemo different

**NERVE** = **N**emo **E**volving **R**easoning via **E**xperience

Your phone agent has a biological nervous system. Not a metaphor — real algorithms inspired by immunology, neuroscience, and evolutionary biology:

| System | Biological Analog | What it does |
|--------|-------------------|--------------|
| 🛡 **Immune System** | White blood cells | Detects anomalous screens (crash dialogs, captchas) and auto-recovers |
| ⚖️ **Homeostasis** | Body temperature regulation | Monitors success rate — slows down when failing, speeds up when confident |
| 🧭 **Entropy Router** | Fast/slow thinking (Kahneman) | Simple tasks use cache (0ms). Complex tasks activate the model (~100ms) |
| 🐜 **Ant Colony (ACO)** | Ant pheromone trails | Learns optimal paths through apps. Routes get stronger with use |
| 🧬 **Genetic Evolution** | Natural selection | Evolves execution strategies over time — your agent gets smarter |
| 🔒 **Safety Invariants** | Pain reflexes | Blocks financial transactions, permission grants, destructive actions |
| 🌙 **Circadian Rhythm** | Sleep/wake cycle | Adapts speed to time of day — fast mornings, gentle evenings |
| 🎯 **Intent Tracker** | Bayesian inference | Handles ambiguous commands by maintaining multiple hypotheses |

**Other AI agents are scripts. Nemo is an organism.**

## Why on-device?

| | Cloud Agents (OpenClaw, etc.) | Nemo |
|---|---|---|
| **Privacy** | Screen data sent to servers | Everything stays on your phone |
| **Cost** | $0.02/step × 100 steps/day = $60/month | Free forever |
| **Speed** | 1-3 seconds per step (network round-trip) | ~100ms per step (on-device) |
| **Offline** | Requires internet | Works on airplane mode |
| **Data** | Your habits on someone's server | Your data, your device |

<a id="demo"></a>
## 🎬 Try it now — Sandbox Mode

**Don't want to set up Accessibility? No problem.**

Nemo includes a built-in Sandbox mode with pre-recorded screen data. Experience the full agent loop without any permissions:

```
Install APK → Open → Tap "Try Sandbox Demo" → Watch the agent work
```

No Accessibility setup. No model download. Just install and tap.

<a id="quickstart"></a>
## ⚡ Quick Start (Full Mode)

### 1. Install
Download the latest APK from [Releases](https://github.com/AShan0227/phone-agent/releases) and install.

### 2. Setup Wizard (3 steps, ~5 minutes)
The app guides you through everything:
- **Step 1:** Enable Accessibility Service (one-tap jump to Settings)
- **Step 2:** Download Gemma 3n E2B model (~1.3GB, automatic, one-time)
- **Step 3:** Try a demo task ("Open Settings")

### 3. Use it
Type any task in natural language. Nemo handles the rest.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                    Nemo Agent                     │
│                                                   │
│  ┌─────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │ NERVE   │  │ Entropy  │  │   On-Device      │  │
│  │ System  │  │ Router   │  │   Gemma 1B       │  │
│  │         │  │          │  │   (MediaPipe)     │  │
│  │ Immune  │  │ System 1 │  │                   │  │
│  │ Homeo.  │  │ Sys. 1.5 │  │   ~100ms/step    │  │
│  │ ACO     │  │ System 2 │  │   529MB model     │  │
│  │ Evol.   │  │          │  │   Zero API cost   │  │
│  └────┬────┘  └────┬─────┘  └────────┬──────────┘  │
│       └────────────┼────────────────┘              │
│                    ▼                                │
│  ┌──────────────────────────────────────────────┐  │
│  │  Screen Understanding                        │  │
│  │  AccessibilityService + OCR + DS Fusion      │  │
│  └──────────────────┬───────────────────────────┘  │
│                     ▼                               │
│  ┌──────────────────────────────────────────────┐  │
│  │  Safety Layer + Privacy Guard                │  │
│  │  Blocks payments, permissions, deletions     │  │
│  │  PII redaction, password field filtering     │  │
│  └──────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
                      │
                      ▼
               Your Android Phone
          (data never leaves device)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| On-device LLM | Gemma 3n E2B via MediaPipe (5B params, 2GB RAM) |
| Screen understanding | AccessibilityService (zero latency) |
| OCR | ML Kit (offline Chinese/English) |
| Knowledge graph | Room + SQLite (A* + ACO) |
| UI | Jetpack Compose |
| Security | AES-256 encrypted storage, HTTPS-only, PII redaction |

## Numbers

| Metric | Value |
|--------|-------|
| Kotlin source | ~5,000 lines |
| Python research | ~10,000 lines |
| Tests | 229 Python + 24 Kotlin |
| Research mechanisms | 15 (all ported to Android) |
| APK size | ~134MB (model downloaded separately) |
| Min Android | 9.0 (API 28) |
| Model | Gemma 3n E2B (5B params, only 2GB RAM, multimodal) |
| Inference speed | ~100ms/step on-device (1.5x faster than Gemma 1B) |

## Research

Nemo implements algorithms from two deep research tracks:

- **Physics-inspired**: Principle of Least Action (A*), Entropy Routing, Dempster-Shafer Fusion, Conservation Laws (Safety), Friction/Inertia (Plan Stability)
- **Biology-inspired**: Negative Selection (Immune), Ant Colony Optimization, Genetic Algorithms, Homeostatic Regulation, Circadian Rhythms, Particle Filter (Intent Tracking)

See `docs/` and `research/` for full papers.

## Privacy & Security

- ✅ All AI inference runs on-device (Gemma 1B via MediaPipe)
- ✅ Screen content never sent to any server
- ✅ Password/PIN/CVV fields automatically filtered from AI context
- ✅ PII (card numbers, ID numbers) automatically redacted
- ✅ Financial transactions, permission grants blocked by default
- ✅ Settings encrypted with AES-256
- ✅ HTTPS-only for model downloads
- ✅ Open source — audit the code yourself

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/AShan0227/phone-agent.git
cd phone-agent
./gradlew assembleDebug   # Build Android app
pytest tests/              # Run Python research tests
```

## License

MIT — use it however you want.

---

<p align="center">
  <b>Other AI agents are scripts. Nemo is an organism.</b><br>
  <sub>Give your phone a nervous system.</sub>
</p>
