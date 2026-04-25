"""
Reward Bridge — Connects ToolCallEnv grading to TRL's reward function format.

TRL's GRPOTrainer expects a reward function that takes completions and returns floats.
This module bridges your existing _grade_easy/medium/hard() into that format.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models import ToolCallAction
from server.environment import ToolCallEnv


def extract_json_from_completion(text: str) -> dict:
    """Extract JSON from model completion text."""
    try:
        return json.loads(text.strip())
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {}


def completion_to_action(completion_text: str, scenario_id: int) -> ToolCallAction:
    """Convert a raw model completion string into a ToolCallAction."""
    parsed = extract_json_from_completion(completion_text)

    return ToolCallAction(
        scenario_id=scenario_id,
        tool_calls=parsed.get("tool_calls", []),
        should_refuse=parsed.get("should_refuse", False),
        reasoning=parsed.get("reasoning", ""),
    )


class RewardBridge:
    """Bridges ToolCallEnv grading to TRL-compatible reward functions."""

    def __init__(self, task_type: str = "hard"):
        self.task_type = task_type
        self.env = ToolCallEnv(task_type=task_type)
        self.env._load_data()

    def grade_completion(self, completion_text: str, scenario: dict) -> float:
        """Grade a single model completion against a scenario."""
        action = completion_to_action(completion_text, scenario["id"])
        try:
            return self.env._grade(action, scenario)
        except Exception:
            return 0.0

    def grade_batch(self, completions: list[str], scenarios: list[dict]) -> list[float]:
        """Grade a batch of completions. Used by TRL's GRPOTrainer."""
        rewards = []
        for comp, scenario in zip(completions, scenarios):
            rewards.append(self.grade_completion(comp, scenario))
        return rewards

    def get_scenarios(self) -> list[dict]:
        """Get all scenarios (without labels — labels stay internal for grading)."""
        return list(self.env.scenarios)

    def get_tools(self) -> list[dict]:
        """Get all tool definitions."""
        return list(self.env.tools)

    def get_scenario_tools(self, scenario: dict) -> list[dict]:
        """Get tool definitions available for a specific scenario."""
        available = scenario.get("available_tools", [])
        return [self.env.tool_lookup[t] for t in available if t in self.env.tool_lookup]


def create_reward_function(task_type: str = "hard"):
    """
    Create a TRL-compatible reward function.

    Returns a callable that takes (completions, prompts, **kwargs) -> list[float]
    The scenarios must be passed via kwargs or pre-bound.
    """
    bridge = RewardBridge(task_type=task_type)

    def reward_fn(completions: list[str], scenarios: Optional[list[dict]] = None, **kwargs) -> list[float]:
        if scenarios is None:
            scenarios = kwargs.get("scenarios", bridge.get_scenarios())

        n = min(len(completions), len(scenarios))
        return bridge.grade_batch(completions[:n], scenarios[:n])

    return reward_fn


def create_grpo_dataset(
    task_type: str = "hard",
    lessons_fn=None,
) -> list[dict]:
    """
    Create a dataset suitable for TRL GRPOTrainer.

    Each entry has a 'prompt' field containing the formatted scenario prompt.
    Optionally enriches prompts with lessons from memory.
    """
    from agent.prompts import build_grpo_prompt

    bridge = RewardBridge(task_type=task_type)
    scenarios = bridge.get_scenarios()
    all_tools = bridge.get_tools()

    dataset = []
    for scenario in scenarios:
        available = scenario.get("available_tools", [])
        tool_defs = [t for t in all_tools if t["name"] in available]

        lessons_text = ""
        if lessons_fn is not None:
            lessons_text = lessons_fn(scenario.get("user_query", ""))

        prompt = build_grpo_prompt(scenario, tool_defs, lessons_text)

        dataset.append({
            "prompt": prompt,
            "scenario_id": scenario["id"],
            "scenario": scenario,
        })

    return dataset
