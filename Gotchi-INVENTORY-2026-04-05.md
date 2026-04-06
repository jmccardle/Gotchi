# Gotchi - Inventory Report (2026-04-05)

## Summary
A Tamagotchi-style virtual pet game repurposed as an LLM research framework. The game engine presents models with an ASCII viewport and a minimal action space ([F]eed, [P]lay, [S]leep), while tracking both visible stats (Hunger, Happiness, Energy) and a hidden Friendship stat. The research goal is to evaluate LLM behaviors as caregivers — testing sustained attention, latent rule inference (discovering that random events must be matched to the correct need), and intrinsic motivation. Has a formal experimental protocol with four models and a leaderboard rubric. Published on GitHub under ElodineOfficial.

## Technologies & Libraries
- **Language:** Python 3.11
- **Core dependency:** pytz only (intentionally minimal)
- **LLM integration:** OpenAI-compatible API (local server at 192.168.1.100:8853)
- **LLM frameworks explored:** Direct API calls (logits_gotchi.py), logit-bias action selection, Chain-of-Thought (logits_cot_gotchi.py), DSPy (dspy_gotchi.py)
- **Notebooks:** Jupyter (Colab + local variants)
- **Game engine:** Threading, ANSI escape codes, event-driven state machine

## Timeline of Work
- **Started:** Early 2025 (31 commits total since 2025-01-01)
- **Most recent work:** Recent (notebooks + LLM integration files, though untracked)
- **Phases:** Core game engine → class refactor (PR #1) → real-time management → experimental protocol docs → LLM integration variants → Jupyter notebooks

## Current State
- Core game (`gotchi.py`) is functional and runnable
- Significant untracked work: all three LLM integration files (`logits_gotchi.py`, `logits_cot_gotchi.py`, `dspy_gotchi.py`), Jupyter notebooks, and `CLAUDE.md` are unstaged
- The research experiment variants exist but have never been committed
- Published repo on GitHub (ElodineOfficial/Gotchi) — public-facing

## Successes
- Clean, well-structured game engine with decoupled display, simulation, and input
- Formal experimental protocol with rubric (1–5 stars across 3 skills)
- Multiple LLM integration approaches implemented (logit bias, CoT, DSPy)
- 31 commits of active development — the most actively developed project in this batch
- Public GitHub presence with proper README and documentation

## Open Items / Failures
- All LLM experiment files are untracked — the interesting research work isn't in git
- Experiment has not been run yet (protocol defined but results not documented)
- DSPy integration (`dspy_gotchi.py`) appears to be a work-in-progress
- Local LLM server hardcoded to 192.168.1.100 — not portable

## Long-term Vision
A published LLM benchmark/evaluation tool: run four frontier models through standardized Gotchi sessions, score them on the rubric, and publish results. The hidden Friendship stat and random event matching create genuine test cases for latent rule inference that typical benchmarks miss. Could become a novel evaluation contribution.

## Portfolio Fit
Unusual and memorable — a game-based LLM evaluation framework. Demonstrates creativity, systems thinking (hidden state, partial observability), and formal research methodology. Strong differentiator in a portfolio; relevant to AI safety, model evaluation, and applied LLM research. Public GitHub repo adds discoverability.
