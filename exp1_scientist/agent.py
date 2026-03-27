"""Experiment 1: LLM Scientist Agent — hypothesis-driven interactive reasoning."""

import json
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from arcengine import GameAction

from shared.perception import serialize_grid, grid_to_ascii
from shared.env_wrapper import ArcEnv, parse_action
from exp1_scientist.config import (
    LOCAL_MODEL, LOCAL_URL, API_KEY, MAX_TOTAL_STEPS,
    CONFIDENCE_THRESHOLD, TEMPERATURE, MAX_TOKENS
)
from exp1_scientist.prompts import SYSTEM_PROMPT
from exp1_scientist.belief import BeliefState


def extract_json(text: str) -> dict:
    """Try to extract JSON from LLM response, handling markdown fences etc."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from code fence
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first { ... }
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


class ScientistAgent:
    """LLM-driven scientist agent that forms and tests hypotheses."""

    def __init__(self, llm_url: str = None, model: str = None):
        self.client = OpenAI(
            base_url=llm_url or LOCAL_URL,
            api_key=API_KEY,
        )
        self.model = model or LOCAL_MODEL
        self.json_parse_successes = 0
        self.json_parse_attempts = 0

    def run_episode(self, game_id: str) -> dict:
        """Run a single episode. Returns result dict."""
        env = ArcEnv(game_id, offline=False, render=False)
        grid, state, score, obs = env.reset()
        belief = BeliefState()
        prev_grid = None
        total_actions = 0
        level = 0

        for step in range(MAX_TOTAL_STEPS):
            structured_obs = serialize_grid(grid, prev_grid)
            ascii_grid = grid_to_ascii(grid)

            # Compact prompt — skip full ASCII grid, use structured data only
            compact_obs = {
                "grid_size": structured_obs["grid_size"],
                "background": structured_obs.get("background_color"),
                "colors": structured_obs["color_counts"],
                "num_objects": structured_obs["num_objects"],
                "delta": structured_obs.get("delta"),
            }
            objects_str = ""
            for obj in structured_obs["objects"][:8]:
                objects_str += f"  color={obj['color']} size={obj['size']} center={obj['center']}\n"

            user_msg = (
                f"Step {step}/{MAX_TOTAL_STEPS} | Actions used: {total_actions}\n"
                f"State: {json.dumps(compact_obs)}\n"
                f"Objects:\n{objects_str}"
            )

            # Tell the LLM which actions are available
            avail_names = [str(a).split(".")[-1] for a in env.available_actions]
            user_msg += f"\n### Available actions: {', '.join(avail_names)}\n"
            user_msg += f"\n{belief.to_prompt_text()}\n\nWhat action should I take?"

            # Query LLM
            decision = self._ask(user_msg)
            self.json_parse_attempts += 1

            if decision is None:
                # Fallback: cycle through available actions
                avail = env.available_actions
                fallback_action = str(avail[step % len(avail)]).split(".")[-1]
                decision = {"action": fallback_action, "reasoning": "JSON parse fallback"}
            else:
                self.json_parse_successes += 1

            # Execute action — constrain to available actions
            action_name = decision.get("action", "ACTION1")
            action = parse_action(action_name)

            # If LLM picked an unavailable action, substitute a valid one
            if action not in env.available_actions:
                action = env.available_actions[step % len(env.available_actions)]

            data = None
            if action == GameAction.ACTION6 and decision.get("x") is not None:
                data = {"x": int(decision["x"]), "y": int(decision["y"])}

            prev_grid = grid.copy()
            grid, state, new_score, obs = env.step(action, data=data)
            total_actions += 1

            # Record effect
            delta = serialize_grid(grid, prev_grid).get("delta", {})
            num_changed = delta.get("num_changed", 0) if delta else 0
            score_delta = new_score - score
            effect_str = f"{num_changed} cells changed, score_delta={score_delta:.2f}"
            belief.log_action(step, action_name, effect_str)
            belief.update_from_llm(decision, step)
            score = new_score

            # Check terminal states
            if state == "WIN":
                level += 1
                return {
                    "won": True, "level": level, "actions": total_actions,
                    "game_id": game_id, "belief": belief.to_dict(),
                    "json_parse_rate": self._parse_rate(),
                }
            if state == "GAME_OVER":
                return {
                    "won": False, "level": level, "actions": total_actions,
                    "game_id": game_id, "belief": belief.to_dict(),
                    "json_parse_rate": self._parse_rate(),
                }

        return {
            "won": False, "level": level, "actions": total_actions,
            "game_id": game_id, "belief": belief.to_dict(),
            "json_parse_rate": self._parse_rate(),
        }

    def _ask(self, user_msg: str) -> dict:
        """Query LLM and parse JSON response."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            text = resp.choices[0].message.content
            return extract_json(text)
        except Exception as e:
            print(f"  [exp1] LLM error: {e}")
            return None

    def _parse_rate(self) -> float:
        if self.json_parse_attempts == 0:
            return 0.0
        return self.json_parse_successes / self.json_parse_attempts

    @property
    def diagnostics(self) -> dict:
        return {
            "json_parse_rate": self._parse_rate(),
            "json_parse_attempts": self.json_parse_attempts,
            "json_parse_successes": self.json_parse_successes,
        }
