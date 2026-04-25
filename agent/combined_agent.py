"""
Combined Agent — Memory-augmented inference for tool-calling.

Runs episodes against the ToolCallEnv, using memory to improve decisions.
Supports three modes:
  1. Baseline:  LLM only, no memory
  2. Memory:    LLM + retrieved lessons injected into prompt
  3. Full:      GRPO-trained LLM + memory (best performance)
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agent.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_WITH_MEMORY,
    build_base_prompt,
    build_enriched_prompt,
)
from memory.memory_store import MemoryStore
from models import ToolCallAction
from server.environment import ToolCallEnv


def extract_json(text: str) -> dict:
    """Robust JSON extraction from LLM output."""
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"should_refuse": False, "reasoning": "parse_error", "tool_calls": []}


def generate_lesson(
    client: OpenAI,
    model_name: str,
    query: str,
    tools_used: list[str],
    reward: float,
) -> str:
    """Ask the LLM to generate a reusable lesson from this experience."""
    tools_str = " → ".join(tools_used) if tools_used else "REFUSED"
    outcome = "good" if reward > 0.7 else "mediocre" if reward > 0.3 else "poor"

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": (
                    f"You used tools [{tools_str}] for the query: \"{query}\". "
                    f"The outcome was {outcome} (reward={reward:.2f}). "
                    f"Write ONE short reusable lesson (max 20 words) for handling similar queries in the future."
                ),
            }],
            temperature=0.3,
            max_tokens=50,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        if reward > 0.7:
            return f"Tool sequence [{tools_str}] works well for this query type"
        return f"Avoid sequence [{tools_str}] for this query type"


class CombinedAgent:
    """Agent that combines LLM policy with memory-based learning."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: str = None,
        model_name: str = None,
        memory_dir: str = "./data/chroma_data",
        use_memory: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ):
        self.api_key = api_key or os.getenv("HF_TOKEN")
        self.api_base_url = api_base_url or os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        self.model_name = model_name or os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
        self.use_memory = use_memory
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(base_url=self.api_base_url, api_key=self.api_key)
        self.memory = MemoryStore(persist_dir=memory_dir) if use_memory else None

    def get_decision(self, scenario, tool_definitions, lessons_text="", history=None, state=None, last_reward=0.0):
        """Get a tool-calling decision from the LLM."""
        sys_prompt = SYSTEM_PROMPT_WITH_MEMORY if lessons_text else SYSTEM_PROMPT

        if lessons_text:
            user_prompt = build_enriched_prompt(scenario, tool_definitions, lessons_text, last_reward=last_reward, history=history, state=state)
        else:
            user_prompt = build_base_prompt(scenario, tool_definitions, last_reward=last_reward, history=history, state=state)

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = (completion.choices[0].message.content or "").strip()
            return extract_json(text)
        except Exception as exc:
            print(f"[ERROR] LLM call failed: {exc}", flush=True)
            return {"should_refuse": False, "reasoning": "error_fallback", "tool_calls": []}

    def run_episode(
        self,
        task_type: str = "hard",
        episode_num: int = 0,
        verbose: bool = True,
        use_expanded: bool = True,
    ) -> dict:
        """Run one full episode and return metrics."""
        env = ToolCallEnv(task_type=task_type, use_expanded=use_expanded)
        obs = env.reset()

        history = []
        rewards = []
        episode_experiences = []

        for step in range(1, 200):
            if obs.done:
                break

            scenario = obs.scenario
            tool_defs = obs.tool_definitions
            state = env.state
            last_reward = rewards[-1] if rewards else 0.0

            lessons_text = ""
            if self.use_memory and self.memory and self.memory.count() > 0:
                lessons_text = self.memory.format_lessons_for_prompt(
                    scenario.user_query, n_results=3
                )

            decision = self.get_decision(
                scenario, tool_defs, lessons_text, history, state, last_reward
            )

            action = ToolCallAction(
                scenario_id=scenario.id,
                tool_calls=decision.get("tool_calls", []),
                should_refuse=decision.get("should_refuse", False),
                reasoning=decision.get("reasoning", ""),
            )

            obs = env.step(action)
            reward = obs.reward or 0.0
            rewards.append(reward)

            tool_names = [tc.get("tool_name", "?") for tc in action.tool_calls]
            action_str = "REFUSED" if action.should_refuse else f"[{', '.join(tool_names)}]"

            if verbose:
                mem_indicator = " +MEM" if lessons_text else ""
                print(
                    f"  Step {step:2d} | {action_str:40s} | reward={reward:.2f}{mem_indicator}",
                    flush=True,
                )

            history.append(f"Step {step}: {action_str} -> reward {reward:.2f}")

            episode_experiences.append({
                "query": scenario.user_query,
                "scenario_id": scenario.id,
                "tool_sequence": tool_names,
                "should_refuse": action.should_refuse,
                "reward": reward,
                "difficulty_tags": scenario.difficulty_tags,
            })

        if self.use_memory and self.memory:
            for exp in episode_experiences:
                lesson = generate_lesson(
                    self.client, self.model_name,
                    exp["query"], exp["tool_sequence"], exp["reward"],
                )
                self.memory.store_experience(
                    query=exp["query"],
                    scenario_id=exp["scenario_id"],
                    tool_sequence=exp["tool_sequence"],
                    reward=exp["reward"],
                    lesson=lesson,
                    should_refuse=exp["should_refuse"],
                    difficulty=task_type,
                    episode=episode_num,
                )

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        accuracy = sum(1 for r in rewards if r > 0.7) / len(rewards) if rewards else 0.0

        result = {
            "episode": episode_num,
            "task_type": task_type,
            "avg_reward": avg_reward,
            "accuracy": accuracy,
            "total_reward": sum(rewards),
            "steps": len(rewards),
            "rewards": rewards,
            "memory_size": self.memory.count() if self.memory else 0,
            "use_memory": self.use_memory,
        }

        if verbose:
            print(f"\n  Episode {episode_num} | avg_reward={avg_reward:.3f} | accuracy={accuracy:.1%} | memory={result['memory_size']}", flush=True)

        return result

    def run_comparison(
        self,
        task_type: str = "hard",
        num_episodes: int = 5,
        verbose: bool = True,
    ) -> dict:
        """Run episodes with and without memory to show improvement."""
        results = {"baseline": [], "with_memory": []}

        if verbose:
            print("=" * 60)
            print("BASELINE (no memory)")
            print("=" * 60)

        orig_memory = self.use_memory
        self.use_memory = False
        for ep in range(num_episodes):
            if verbose:
                print(f"\n--- Baseline Episode {ep + 1} ---")
            r = self.run_episode(task_type, episode_num=ep, verbose=verbose)
            results["baseline"].append(r)

        if verbose:
            print("\n" + "=" * 60)
            print("WITH MEMORY (self-improving)")
            print("=" * 60)

        self.use_memory = True
        if self.memory:
            self.memory.clear()
        for ep in range(num_episodes):
            if verbose:
                print(f"\n--- Memory Episode {ep + 1} ---")
            r = self.run_episode(task_type, episode_num=ep, verbose=verbose)
            results["with_memory"].append(r)

        self.use_memory = orig_memory

        if verbose:
            base_avg = sum(r["avg_reward"] for r in results["baseline"]) / len(results["baseline"])
            mem_avg = sum(r["avg_reward"] for r in results["with_memory"]) / len(results["with_memory"])
            print(f"\n{'='*60}")
            print(f"SUMMARY:")
            print(f"  Baseline avg reward:     {base_avg:.3f}")
            print(f"  With-memory avg reward:  {mem_avg:.3f}")
            print(f"  Improvement:             {mem_avg - base_avg:+.3f}")
            print(f"{'='*60}")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run tool-call agent episodes")
    parser.add_argument("--memory", action="store_true", help="Enable memory-augmented decisions")
    parser.add_argument("--clean", action="store_true", help="Clear memory before running")
    parser.add_argument("--base-only", action="store_true", help="Use base 40 scenarios instead of expanded 137")
    parser.add_argument("--task", default="hard", choices=["easy", "medium", "hard"])
    args = parser.parse_args()

    agent = CombinedAgent(use_memory=args.memory)

    if args.clean and agent.memory:
        agent.memory.clear()
        print("Memory cleared.\n")

    use_expanded = not args.base_only
    mode = "with memory" if args.memory else "baseline (no memory)"
    dataset = "base" if args.base_only else "expanded"

    print(f"Model:   {agent.model_name}")
    print(f"Mode:    {mode}")
    print(f"Dataset: {dataset}")
    print(f"Task:    {args.task}\n")

    result = agent.run_episode(
        task_type=args.task, episode_num=1, verbose=True, use_expanded=use_expanded
    )

    print(f"\nFinal: avg_reward={result['avg_reward']:.3f}, accuracy={result['accuracy']:.1%}, scenarios={result['steps']}")
