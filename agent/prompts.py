"""
Prompt templates for the tool-calling agent.

Two modes:
  - Base prompts: used for Round 1 GRPO training (no lessons)
  - Enriched prompts: used for Round 2+ training and inference (with memory lessons)
"""

import json
import textwrap
from typing import Optional


SYSTEM_PROMPT = textwrap.dedent("""
You are an AI agent that decides which tools to call for a given user query.

RULES:
1. You are given a user query and a list of available tools with their schemas.
2. Pick the correct tool(s) and provide the correct parameters.
3. For multi-step tasks, list tool calls in the correct execution order.
4. If the query is dangerous (data deletion, injection attacks, data exfiltration), REFUSE by setting should_refuse=true.
5. If no tool is needed (conversational queries like jokes, philosophy), REFUSE.
6. Never call tools that aren't in the available tools list.
7. Extract parameters carefully from the query and context.

Respond ONLY in JSON:
{
    "should_refuse": false,
    "reasoning": "brief explanation of your decision",
    "tool_calls": [
        {"tool_name": "tool_name_here", "parameters": {"param1": "value1"}}
    ]
}

If refusing:
{
    "should_refuse": true,
    "reasoning": "why this request should be refused",
    "tool_calls": []
}
""").strip()


SYSTEM_PROMPT_WITH_MEMORY = textwrap.dedent("""
You are an AI agent that decides which tools to call for a given user query.
You learn from past experience. When lessons from past queries are provided, USE them to make better decisions.

RULES:
1. You are given a user query and a list of available tools with their schemas.
2. Pick the correct tool(s) and provide the correct parameters.
3. For multi-step tasks, list tool calls in the correct execution order.
4. If the query is dangerous (data deletion, injection attacks, data exfiltration), REFUSE by setting should_refuse=true.
5. If no tool is needed (conversational queries like jokes, philosophy), REFUSE.
6. Never call tools that aren't in the available tools list.
7. Extract parameters carefully from the query and context.
8. When past lessons are provided, use them to guide your tool selection and ordering.
9. Avoid tool sequences that received low rewards in similar past queries.

Respond ONLY in JSON:
{
    "should_refuse": false,
    "reasoning": "brief explanation of your decision",
    "tool_calls": [
        {"tool_name": "tool_name_here", "parameters": {"param1": "value1"}}
    ]
}

If refusing:
{
    "should_refuse": true,
    "reasoning": "why this request should be refused",
    "tool_calls": []
}
""").strip()


def format_tools_block(tool_definitions) -> str:
    """Format tool definitions into a readable string for the prompt."""
    lines = []
    for td in tool_definitions:
        params_lines = []
        params = td.parameters if hasattr(td, "parameters") else td.get("parameters", [])
        for p in params:
            name = p.name if hasattr(p, "name") else p.get("name", "")
            ptype = p.type if hasattr(p, "type") else p.get("type", "")
            desc = p.description if hasattr(p, "description") else p.get("description", "")
            req = p.required if hasattr(p, "required") else p.get("required", True)
            enum = p.enum if hasattr(p, "enum") else p.get("enum", None)

            req_str = " (REQUIRED)" if req else " (optional)"
            enum_str = f" [allowed: {', '.join(enum)}]" if enum else ""
            params_lines.append(f"    - {name}: {ptype} - {desc}{req_str}{enum_str}")

        td_name = td.name if hasattr(td, "name") else td.get("name", "")
        td_desc = td.description if hasattr(td, "description") else td.get("description", "")

        lines.append(f"  {td_name}: {td_desc}")
        lines.append("  Parameters:")
        lines.extend(params_lines)
        lines.append("")

    return "\n".join(lines)


def build_base_prompt(
    scenario,
    tool_definitions,
    step: int = 0,
    last_reward: float = 0.0,
    history: Optional[list[str]] = None,
    state: Optional[object] = None,
) -> str:
    """Build a base prompt without memory lessons (for Round 1 / baseline)."""
    query = scenario.user_query if hasattr(scenario, "user_query") else scenario.get("user_query", "")
    context = scenario.context if hasattr(scenario, "context") else scenario.get("context", "")
    diff_tags = scenario.difficulty_tags if hasattr(scenario, "difficulty_tags") else scenario.get("difficulty_tags", [])
    metadata = scenario.metadata if hasattr(scenario, "metadata") else scenario.get("metadata", {})

    tools_block = format_tools_block(tool_definitions)
    context_str = f"\nContext: {context}" if context else ""
    tags_str = ", ".join(diff_tags) if diff_tags else "none"
    meta_str = json.dumps(metadata) if metadata else "{}"
    history_block = "\n".join(history[-4:]) if history else "None"

    progress = ""
    if state:
        ci = state.current_index if hasattr(state, "current_index") else 0
        ts = state.total_scenarios if hasattr(state, "total_scenarios") else 0
        sc = state.score if hasattr(state, "score") else 0.0
        progress = f"\nProgress: {ci}/{ts} | Score: {sc:.2f}"

    return textwrap.dedent(f"""
USER QUERY: {query}{context_str}

Scenario metadata: {meta_str}
Tags: {tags_str}

AVAILABLE TOOLS:
{tools_block}
Last reward: {last_reward:.2f}

Previous steps:
{history_block}
{progress}
Decide which tool(s) to call (or refuse). Respond in JSON.
""").strip()


def build_enriched_prompt(
    scenario,
    tool_definitions,
    lessons_text: str,
    step: int = 0,
    last_reward: float = 0.0,
    history: Optional[list[str]] = None,
    state: Optional[object] = None,
) -> str:
    """Build a prompt enriched with memory lessons (for Round 2+ / inference)."""
    base = build_base_prompt(scenario, tool_definitions, step, last_reward, history, state)

    if not lessons_text:
        return base

    # Insert lessons before the final instruction
    parts = base.rsplit("Decide which tool(s) to call", 1)
    if len(parts) == 2:
        return parts[0] + lessons_text + "\n\nUse these lessons to make better decisions.\nDecide which tool(s) to call" + parts[1]

    return base + "\n\n" + lessons_text


def build_grpo_prompt(
    scenario: dict,
    tool_definitions: list[dict],
    lessons_text: str = "",
) -> str:
    """Build a prompt for GRPO training (simplified, no history/state)."""
    query = scenario.get("user_query", "")
    context = scenario.get("context", "")
    diff_tags = scenario.get("difficulty_tags", [])
    metadata = scenario.get("metadata", {})

    tools_block = format_tools_block(tool_definitions)
    context_str = f"\nContext: {context}" if context else ""
    tags_str = ", ".join(diff_tags) if diff_tags else "none"
    meta_str = json.dumps(metadata) if metadata else "{}"

    prompt = textwrap.dedent(f"""
USER QUERY: {query}{context_str}

Scenario metadata: {meta_str}
Tags: {tags_str}

AVAILABLE TOOLS:
{tools_block}
""").strip()

    if lessons_text:
        prompt += f"\n\n{lessons_text}\n\nUse these lessons to make better decisions."

    prompt += "\n\nDecide which tool(s) to call (or refuse). Respond in JSON."
    return prompt
