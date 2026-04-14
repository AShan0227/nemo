# Contributing to Nemo

Thanks for your interest! Nemo is an open-source on-device phone agent — we welcome all contributions.

## Getting Started

```bash
git clone https://github.com/AShan0227/phone-agent.git
cd phone-agent

# Android app
./gradlew assembleDebug        # Build APK
./gradlew test                  # Run Kotlin unit tests

# Python research
pip install -e ".[dev]"
pytest tests/                   # Run Python tests
```

## Project Structure

```
app/src/main/java/com/nemo/   ← Android Kotlin app (production)
├── agent/                      ← Core agent loop + planner + action builder
├── model/                      ← On-device LLM + entropy router + prompts
├── screen/                     ← Screen understanding + OCR + fusion
├── knowledge/                  ← Knowledge graph + genome DSL
├── safety/                     ← Safety layer + privacy guard
├── research/                   ← 8 bio-inspired research mechanisms
├── service/                    ← AccessibilityService + foreground service
└── ui/                         ← Jetpack Compose UI

src/                           ← Python research code (algorithm reference)
research/                      ← Research papers
```

## How to Contribute

### Good First Issues
Look for issues labeled `good-first-issue` — these are beginner-friendly.

### Areas We Need Help
- **Real device testing** — test on different Android phones and report issues
- **Chinese NLP** — improve Chinese language understanding in prompts
- **App adapters** — add support for specific apps (WeChat, Alipay, etc.)
- **Model optimization** — quantization, pruning, faster inference
- **Documentation** — tutorials, blog posts, translation

### Pull Request Process
1. Fork the repo and create a branch (`feature/your-feature`)
2. Make your changes
3. Run tests: `./gradlew test && pytest tests/`
4. Submit a PR with a clear description

### Code Style
- Kotlin: follow [Kotlin coding conventions](https://kotlinlang.org/docs/coding-conventions.html)
- Python: `ruff` for linting, `mypy` for type checking
- No comments unless the WHY is non-obvious

## Research Mechanisms

If you want to contribute to the NERVE system (biology-inspired algorithms), read:
- `docs/physics-inspired-mechanisms-research.md`
- `research/biology_inspired_mechanisms.md`

Each mechanism in `app/src/main/java/com/nemo/research/` is self-contained and can be improved independently.

## License

MIT — your contributions will be under the same license.
