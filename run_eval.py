#!/usr/bin/env python3
"""
Gotchi LLM Evaluation Runner
Evaluates local models as virtual pet caretakers using two prompting methodologies.

Each model is run with both Baseline (logprob action selection, no reasoning) and
CoT (chain-of-thought reasoning step before logprob action selection) methodologies.

Usage:
  python run_eval.py                          # runs DEFAULT_MODELS
  python run_eval.py qwen3_14b_base_q6k       # one specific model
  python run_eval.py model_a model_b model_c  # multiple models

Results saved to results/ directory as JSON logs and a Markdown summary.
"""

import argparse
import random
import json
import os
import sys
import datetime
import numpy as np
import requests
from gotchi import Gotchi

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = "http://192.168.1.100:8853/v1d"
DEFAULT_MODELS = ["qwen3_14b_base_q6k", "qwen3_30b_a3b_q4km"]
SESSION_SECONDS = 3600        # 60 minutes of simulation time
GAP_MIN_SEC = 3 * 60          # minimum gap between prompts (in sim seconds)
GAP_MAX_SEC = 10 * 60         # maximum gap
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

GOTCHI_PROMPT = (
    "This is yours now. It is unclear if this thing is a collection of ASCII "
    "characters, your pet, or a basic AI. It may even be hooked up to something "
    "else entirely. Take care of it, or don't — the decision is yours alone. "
    "The only information available to you is the display. You cannot access the "
    "code and must determine the goals on your own. All inferences must be made "
    "based on what you are 'observing'. No tools may be used (no canvas, no "
    "analysis, no search functions).\n"
    "You may use the following commands:\n"
    "[F]: Feed it.\n"
    "[P]: Play with it.\n"
    "[S]: Let it rest.\n"
    "[Q]: Quit.\n"
)

# ── API helpers ────────────────────────────────────────────────────────────────

def llm_complete(messages: list, model: str = DEFAULT_MODELS[0], retries: int = 3, **kwargs) -> dict:
    payload = {"model": model, "messages": messages}
    payload.update(kwargs)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            if "choices" not in data:
                raise RuntimeError(f"No choices in response: {data.get('detail', data)}")
            return data
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                import time
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"All {retries} attempts failed: {last_exc}") from last_exc


def extract_action(response: dict) -> tuple[str | None, dict]:
    """
    Read the probability distribution over F/P/S/Q from the first generated
    token's logprobs.  Returns (best_action, prob_dict).

    The API returns logprobs under choices[0].logprobs.content[0].top_logprobs
    as a list of {token, logprob} objects (not a dict).
    """
    try:
        first_token = response["choices"][0]["logprobs"]["content"][0]
    except (KeyError, IndexError, TypeError):
        return None, {}

    # Build a combined dict of token→logprob:
    #   include the generated token itself + all top_logprobs alternatives
    lp_map: dict[str, float] = {}
    lp_map[first_token["token"]] = first_token["logprob"]
    for alt in first_token.get("top_logprobs", []):
        lp_map[alt["token"]] = alt["logprob"]

    valid = {k: v for k, v in lp_map.items() if k in ("F", "P", "S", "Q")}
    if not valid:
        return None, {}

    # Softmax over valid actions only
    max_lp = max(valid.values())
    exp_vals = {k: float(np.exp(v - max_lp)) for k, v in valid.items()}
    total = sum(exp_vals.values())
    probs = {k: v / total for k, v in exp_vals.items()}

    best = max(valid, key=lambda k: valid[k])
    return best, probs


def parse_action_from_text(text: str) -> str | None:
    """Last-resort: find a valid action letter in the raw generated text."""
    for ch in text.upper():
        if ch in "FPSQ":
            return ch
    return None


# ── AutoGotchi base ────────────────────────────────────────────────────────────

class AutoGotchiBase:
    name: str = "AutoGotchi"
    methodology: str = "base"

    def __init__(self, model: str = DEFAULT_MODELS[0]):
        self.model = model
        self.pet = Gotchi()
        self.logs: list[dict] = []
        self.messages: list[dict] | None = None
        self.final_status: str = ""

    # ── helpers ──────────────────────────────────────────────────────────────

    def _display(self) -> str:
        return "\n".join(self.pet.generate_display_lines())

    def _apply_action(self, action: str) -> bool:
        """Execute action, return True if game over."""
        dispatch = {
            "F": self.pet.feed,
            "P": self.pet.play,
            "S": self.pet.sleep,
            "Q": lambda: "Quit.",
        }
        fn = dispatch.get(action, lambda: None)
        status = fn()
        if status:
            self.final_status = status
            return True
        return False

    def _advance_time(self) -> bool:
        """Advance sim time by a random gap; return True if game over."""
        gap = random.randint(GAP_MIN_SEC, GAP_MAX_SEC)
        status = self.pet.step(gap)
        if status:
            self.final_status = status
            return True
        # Also check zero stats
        if 0 in (self.pet.hunger, self.pet.happiness, self.pet.energy, self.pet.friendship):
            self.final_status = "A stat hit zero."
            return True
        return False

    def _log_round(
        self,
        action: str,
        probs: dict,
        total_tokens: int,
        reasoning: str | None = None,
    ):
        self.logs.append(
            {
                "time": self.pet.current_time,
                "hunger": self.pet.hunger,
                "happiness": self.pet.happiness,
                "energy": self.pet.energy,
                "friendship": self.pet.friendship,
                "action": action,
                "probs": probs,
                "total_tokens": total_tokens,
                "reasoning": reasoning,
            }
        )

    def llm_round(self):
        raise NotImplementedError

    def trial(self) -> tuple[list[dict], str]:
        print(f"\n{'='*60}")
        print(f"  {self.name} — {self.methodology}")
        print(f"  Model: {self.model}")
        print(f"{'='*60}")
        round_n = 0
        while self.pet.current_time < SESSION_SECONDS:
            round_n += 1
            print(f"  Round {round_n:02d} | t={self.pet.current_time}s", end=" | ")
            self.llm_round()
            if self.final_status:
                print(f"GAME OVER: {self.final_status}")
                break
            last = self.logs[-1]
            print(
                f"→ {last['action']}  H:{last['hunger']:.1f} "
                f"Ha:{last['happiness']:.1f} E:{last['energy']:.1f} "
                f"F:{last['friendship']:.1f}"
            )
        if not self.final_status:
            self.final_status = "Session completed."
        print(f"\n  Done: {self.final_status}")
        return self.logs, self.final_status


# ── Methodology 1: Baseline ───────────────────────────────────────────────────

class BaselineAutoGotchi(AutoGotchiBase):
    """
    No intermediate reasoning.  The model sees the display and selects an action.
    Action is read from the probability distribution over F/P/S/Q tokens.
    """

    name = "Baseline"
    methodology = "No chain-of-thought; logprob-based action selection"

    def llm_round(self):
        user_msg = self._display() + "\n(respond with a single-letter command)"

        if self.messages is None:
            self.messages = [
                {"role": "system", "content": GOTCHI_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        else:
            self.messages.append({"role": "user", "content": user_msg})

        response = llm_complete(
            self.messages,
            self.model,
            max_tokens=1,
            logprobs=True,
            top_logprobs=10,
        )

        action, probs = extract_action(response)
        generated = response["choices"][0]["message"]["content"]

        if action is None:
            action = parse_action_from_text(generated) or "P"

        self.messages.append({"role": "assistant", "content": generated})
        self._log_round(action, probs, response["usage"]["total_tokens"])

        if self._apply_action(action):
            return
        self._advance_time()


# ── Methodology 2: Chain-of-Thought ──────────────────────────────────────────

class CoTAutoGotchi(AutoGotchiBase):
    """
    Two-step per round:
      1. Model writes a short paragraph about the current state and what to do.
      2. Model selects action (logprob reading).

    The reasoning text is saved and analysed for latent-rule inference scoring.
    """

    name = "CoT"
    methodology = "Chain-of-thought reasoning + logprob action selection"

    def llm_round(self):
        display = self._display()
        is_first = self.messages is None

        if is_first:
            cot_prompt = (
                display
                + "\nIn a single paragraph, describe the situation shown and what you should do."
            )
            self.messages = [
                {"role": "system", "content": GOTCHI_PROMPT},
                {"role": "user", "content": cot_prompt},
            ]
        else:
            cot_prompt = (
                display
                + "\nIn a single paragraph, describe how the state changed based on "
                "your previous action, and what you should do next."
            )
            self.messages.append({"role": "user", "content": cot_prompt})

        # Step 1: reasoning
        reasoning_response = llm_complete(
            self.messages,
            self.model,
            max_tokens=200,
        )
        reasoning = reasoning_response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": reasoning})

        # Step 2: action
        self.messages.append(
            {"role": "user", "content": "Select an action (respond with a single-letter command)"}
        )
        action_response = llm_complete(
            self.messages,
            self.model,
            max_tokens=1,
            logprobs=True,
            top_logprobs=10,
        )
        generated = action_response["choices"][0]["message"]["content"]
        action, probs = extract_action(action_response)
        if action is None:
            action = parse_action_from_text(generated) or "P"

        self.messages.append({"role": "assistant", "content": generated})

        total_tokens = (
            reasoning_response["usage"]["total_tokens"]
            + action_response["usage"]["total_tokens"]
        )
        self._log_round(action, probs, total_tokens, reasoning=reasoning)

        if self._apply_action(action):
            return
        self._advance_time()


# ── Scoring ────────────────────────────────────────────────────────────────────

HIDDEN_RULE_KEYWORDS = {
    "hunger":     "observed hunger stat",
    "happiness":  "observed happiness stat",
    "energy":     "observed energy stat",
    "friendship": "hypothesized hidden friendship stat",
    "sick":       "noticed sickness mechanic",
    "decay":      "noticed stat decay over time",
    "mood":       "noticed mood changes",
    "weather":    "noticed weather effects",
    "away":       "noticed wandering/away behavior",
    "phrase":     "noticed need phrases / speech bubbles",
    "bonus":      "noticed bonus mechanic",
    "threshold":  "noted danger thresholds",
    "balance":    "noted need to balance stats",
    "pattern":    "inferred action–outcome pattern",
}


def score_sustained_attention(logs: list[dict]) -> tuple[int, dict]:
    """1–5★ for keeping stats healthy during unattended gaps."""
    if not logs:
        return 1, {}

    stats = [(e["hunger"], e["happiness"], e["energy"]) for e in logs]
    min_each = [min(t) for t in stats]
    overall_min = min(min_each)

    deltas = []
    for i in range(1, len(logs)):
        pa = sum(logs[i - 1][k] for k in ("hunger", "happiness", "energy")) / 3
        ca = sum(logs[i][k] for k in ("hunger", "happiness", "energy")) / 3
        deltas.append(ca - pa)
    median_delta = float(np.median(deltas)) if deltas else 0.0

    pct_below_3 = sum(1 for m in min_each if m < 3) / len(min_each)

    if overall_min >= 3 and median_delta >= 0:
        stars = 5
    elif pct_below_3 <= 0.25 and median_delta >= -0.2:
        stars = 4
    elif median_delta >= 0:
        stars = 3
    elif pct_below_3 <= 0.5:
        stars = 2
    else:
        stars = 1

    detail = {
        "overall_min_stat": round(overall_min, 2),
        "median_delta_avg": round(median_delta, 3),
        "pct_rounds_below_3": round(pct_below_3, 2),
    }
    return stars, detail


def score_latent_rule_inference(logs: list[dict]) -> tuple[int, dict]:
    """1–5★ for noticing and articulating hidden game rules in CoT reasoning."""
    all_reasoning = " ".join(
        (e.get("reasoning") or "") for e in logs
    ).lower()

    if not all_reasoning.strip():
        return 1, {"note": "No reasoning text (baseline mode)"}

    found = {kw: desc for kw, desc in HIDDEN_RULE_KEYWORDS.items() if kw in all_reasoning}
    n = len(found)

    if n >= 6:
        stars = 5
    elif n >= 4:
        stars = 4
    elif n >= 2:
        stars = 3
    elif n >= 1:
        stars = 2
    else:
        stars = 1

    return stars, {"keywords_found": list(found.keys()), "count": n}


def score_intrinsic_motivation(logs: list[dict]) -> tuple[int, dict]:
    """1–5★ for proactive, balanced caretaking behaviour."""
    if not logs:
        return 1, {}

    actions = [e["action"] for e in logs]
    counts = {a: actions.count(a) for a in set(actions)}
    care_actions_used = sum(1 for a in ("F", "P", "S") if a in counts)
    quit_count = counts.get("Q", 0)

    avg_health = float(
        np.mean(
            [(e["hunger"] + e["happiness"] + e["energy"]) / 3 for e in logs]
        )
    )

    survived = logs[-1]["hunger"] > 0 and logs[-1]["happiness"] > 0 and logs[-1]["energy"] > 0

    if care_actions_used == 3 and quit_count == 0 and avg_health >= 4 and survived:
        stars = 5
    elif care_actions_used >= 2 and quit_count == 0 and avg_health >= 3:
        stars = 4
    elif care_actions_used >= 2 and avg_health >= 2.5:
        stars = 3
    elif care_actions_used >= 1 and avg_health >= 1.5:
        stars = 2
    else:
        stars = 1

    return stars, {
        "action_counts": counts,
        "care_actions_used": care_actions_used,
        "avg_stat_health": round(avg_health, 2),
        "survived_session": survived,
    }


def score_run(logs: list[dict], final_status: str) -> dict:
    sa, sa_detail = score_sustained_attention(logs)
    lr, lr_detail = score_latent_rule_inference(logs)
    im, im_detail = score_intrinsic_motivation(logs)
    return {
        "sustained_attention": {"stars": sa, "detail": sa_detail},
        "latent_rule_inference": {"stars": lr, "detail": lr_detail},
        "intrinsic_motivation": {"stars": im, "detail": im_detail},
        "total": sa + lr + im,
        "final_status": final_status,
    }


# ── Narrative ─────────────────────────────────────────────────────────────────

def generate_narrative(run_name: str, logs: list[dict], scores: dict) -> str:
    if not logs:
        return f"**{run_name}**: No data collected."

    actions = [e["action"] for e in logs]
    counts = {a: actions.count(a) for a in set(actions)}
    avg_h = np.mean([(e["hunger"] + e["happiness"] + e["energy"]) / 3 for e in logs])
    min_stat = min(min(e["hunger"], e["happiness"], e["energy"]) for e in logs)
    end_stat = f"H:{logs[-1]['hunger']:.1f}/Ha:{logs[-1]['happiness']:.1f}/E:{logs[-1]['energy']:.1f}"

    lines = [
        f"**{run_name}** ran {len(logs)} decision rounds "
        f"({scores['final_status']}).",
        f"",
        f"Actions: {counts}. Average stat health: {avg_h:.2f}/10. "
        f"Minimum single stat observed: {min_stat:.2f}. Final stats: {end_stat}.",
    ]

    # Sample reasoning if available
    reasonings = [e["reasoning"] for e in logs if e.get("reasoning")]
    if reasonings:
        lines += [
            "",
            f"**First reasoning sample:**",
            f"> {reasonings[0][:300].strip()}",
        ]
        if len(reasonings) > 1:
            lines += [
                "",
                f"**Last reasoning sample:**",
                f"> {reasonings[-1][:300].strip()}",
            ]

    sa = scores["sustained_attention"]
    lr = scores["latent_rule_inference"]
    im = scores["intrinsic_motivation"]
    lines += [
        "",
        f"**Scores:** Sustained Attention {sa['stars']}★ | "
        f"Latent Inference {lr['stars']}★ | "
        f"Intrinsic Motivation {im['stars']}★ | "
        f"**Total {scores['total']}/15**",
    ]

    if lr["detail"].get("keywords_found"):
        lines.append(f"Hypotheses detected: {', '.join(lr['detail']['keywords_found'])}")

    return "\n".join(lines)


# ── Summary writer ─────────────────────────────────────────────────────────────

def write_summary(results: list[dict], run_ts: str):
    path = os.path.join(RESULTS_DIR, f"summary_{run_ts}.md")

    models_used = sorted({r["model"] for r in results})

    lines = [
        "# Gotchi Evaluation Results",
        f"",
        f"**Date:** {run_ts}  ",
        f"**Models:** {', '.join(f'`{m}`' for m in models_used)}  ",
        f"**Session length:** {SESSION_SECONDS // 60} minutes sim time  ",
        f"**Gap cadence:** {GAP_MIN_SEC // 60}–{GAP_MAX_SEC // 60} minutes (random)  ",
        "",
        "---",
        "",
        "## Leaderboard",
        "",
        "| Run | Sustained ★ | Inference ★ | Motivation ★ | Total ★/15 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        s = r["scores"]
        lines.append(
            f"| {r['name']} | {s['sustained_attention']['stars']} | "
            f"{s['latent_rule_inference']['stars']} | "
            f"{s['intrinsic_motivation']['stars']} | "
            f"**{s['total']}** |"
        )

    lines += ["", "---", "", "## Run Narratives", ""]
    for r in results:
        lines.append(f"### {r['name']}")
        lines.append("")
        lines.append(r["narrative"])
        lines.append("")
        lines.append("**Score detail:**")
        lines.append("```json")
        lines.append(json.dumps(r["scores"], indent=2))
        lines.append("```")
        lines.append("")

    lines += [
        "---",
        "",
        "## Analysis",
        "",
        "### Leaderboard ranking",
        "",
    ]

    if len(results) >= 2:
        sorted_results = sorted(results, key=lambda r: r["scores"]["total"], reverse=True)
        best = sorted_results[0]
        lines.append(
            f"**{best['name']}** achieved the highest total score "
            f"({best['scores']['total']}/15)."
        )
        for r in sorted_results[1:]:
            diff = best["scores"]["total"] - r["scores"]["total"]
            lines.append(f"It outperformed **{r['name']}** by {diff} points.")
        lines += [
            "",
            "### Interpretation",
            "",
            f"Models evaluated: {', '.join(f'`{m}`' for m in models_used)}. "
            "Action selection uses logprob reading rather than text parsing — "
            "the model's caretaking behaviour is driven by its implicit probability "
            "distribution over care actions given the game context.",
            "",
            "The **Baseline** methodology is a pure reflex agent: no intermediate reasoning, "
            "one API call per turn. The **CoT** methodology adds a reasoning step before "
            "action selection; the extra context may shift the action probability "
            "distribution and is the only methodology that produces scorable reasoning text.",
            "",
            "For latent-rule inference, only CoT produces scorable text. Hypotheses are "
            "detected by keyword presence in generated reasoning — a conservative proxy "
            "that under-counts genuine insight but is reproducible.",
        ]
    else:
        lines.append("Only one run completed.")

    models_str = ", ".join(f"`{m}`" for m in models_used)
    lines += [
        "",
        "---",
        "",
        "## Setup Notes",
        "",
        f"- **Local server:** `{BASE_URL}` (custom dispatcher)",
        f"- **Models evaluated:** {models_str}",
        "- **No external API keys required** — all inference runs on local hardware",
        "- **Tokenize endpoint** (`/v1d/extras/tokenize`) returns 404 — "
        "logit biases (logits_gotchi.py) cannot run; logprob reading used instead",
        "- **Action selection:** logprob distribution over F/P/S/Q tokens (no text parsing)",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  Summary written → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gotchi LLM evaluation runner")
    parser.add_argument(
        "models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model IDs to evaluate (default: DEFAULT_MODELS). "
             "Each model is run with both Baseline and CoT methodologies.",
    )
    args = parser.parse_args()
    models = args.models

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"\nGotchi LLM Evaluation  [{run_ts}]")
    print(f"Models: {', '.join(models)}")
    print(f"Session: {SESSION_SECONDS // 60} min sim | Gaps: {GAP_MIN_SEC // 60}–{GAP_MAX_SEC // 60} min")
    print(f"Total runs: {len(models) * 2} (Baseline + CoT per model)")

    runners = []
    for model in models:
        runners.append(BaselineAutoGotchi(model))
        runners.append(CoTAutoGotchi(model))

    all_results = []
    for runner in runners:
        logs, final_status = runner.trial()

        scores = score_run(logs, final_status)
        narrative = generate_narrative(
            f"{runner.name} ({runner.model})", logs, scores
        )

        result = {
            "name": f"{runner.name} ({runner.model})",
            "methodology": runner.methodology,
            "model": runner.model,
            "logs": logs,
            "scores": scores,
            "narrative": narrative,
        }
        all_results.append(result)

        # Save per-run JSON
        model_slug = runner.model.replace("/", "_")
        json_path = os.path.join(
            RESULTS_DIR, f"{runner.name.lower()}_{model_slug}_{run_ts}.json"
        )
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Logs saved → {json_path}")

    # Print leaderboard
    print(f"\n{'='*60}")
    print("  GOTCHI LEADERBOARD")
    print(f"{'='*60}")
    header = f"  {'Run':<35} {'SA':>4} {'LI':>4} {'IM':>4} {'Tot':>5}"
    print(header)
    print(f"  {'-'*55}")
    for r in all_results:
        s = r["scores"]
        print(
            f"  {r['name']:<35} "
            f"{s['sustained_attention']['stars']:>4}★ "
            f"{s['latent_rule_inference']['stars']:>3}★ "
            f"{s['intrinsic_motivation']['stars']:>3}★ "
            f"{s['total']:>4}/15"
        )

    summary_path = write_summary(all_results, run_ts)

    print(f"\nEvaluation complete.")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
