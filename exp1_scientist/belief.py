"""Belief state tracking for hypothesis management."""

import json


class BeliefState:
    def __init__(self):
        self.action_hypotheses = {}  # action -> list of {rule, confidence, evidence}
        self.goal_hypotheses = []    # list of {description, confidence, evidence}
        self.action_log = []         # list of {step, action, effect}
        self.max_confidence = 0.0

    def update_from_llm(self, decision: dict, step: int):
        """Update beliefs from LLM response."""
        uh = decision.get("updated_hypotheses", {})

        for ae in uh.get("action_effects", []):
            action = ae.get("action", "unknown")
            self.action_hypotheses.setdefault(action, [])
            found = False
            for existing in self.action_hypotheses[action]:
                if existing["rule"] == ae.get("rule", ""):
                    existing["confidence"] = ae.get("confidence", 0.5)
                    existing["evidence"] = f"step {step}"
                    found = True
                    break
            if not found and ae.get("rule"):
                self.action_hypotheses[action].append({
                    "rule": ae["rule"],
                    "confidence": ae.get("confidence", 0.5),
                    "evidence": f"step {step}",
                })

        for g in uh.get("goals", []):
            if g.get("description"):
                found = False
                for existing in self.goal_hypotheses:
                    if existing["description"] == g["description"]:
                        existing["confidence"] = g.get("confidence", 0.5)
                        existing["evidence"] = f"step {step}"
                        found = True
                        break
                if not found:
                    self.goal_hypotheses.append({
                        "description": g["description"],
                        "confidence": g.get("confidence", 0.5),
                        "evidence": f"step {step}",
                    })

        # Track max confidence for phase switching
        all_confs = [h["confidence"] for hyps in self.action_hypotheses.values() for h in hyps]
        all_confs += [g["confidence"] for g in self.goal_hypotheses]
        self.max_confidence = max(all_confs) if all_confs else 0.0

    def log_action(self, step: int, action: str, effect: str):
        self.action_log.append({"step": step, "action": action, "effect": effect})

    def to_prompt_text(self) -> str:
        lines = ["## Current Beliefs"]

        if self.action_hypotheses:
            lines.append("### Action effects:")
            for action, hyps in self.action_hypotheses.items():
                for h in hyps:
                    lines.append(
                        f"  - {action}: \"{h['rule']}\" "
                        f"(confidence: {h['confidence']:.0%})"
                    )

        if self.goal_hypotheses:
            lines.append("### Goal hypotheses:")
            for h in self.goal_hypotheses:
                lines.append(
                    f"  - \"{h['description']}\" "
                    f"(confidence: {h['confidence']:.0%})"
                )

        lines.append(f"### Actions taken: {len(self.action_log)}")
        for entry in self.action_log[-8:]:
            lines.append(f"  Step {entry['step']}: {entry['action']} -> {entry['effect']}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "action_hypotheses": self.action_hypotheses,
            "goal_hypotheses": self.goal_hypotheses,
            "max_confidence": self.max_confidence,
        }
