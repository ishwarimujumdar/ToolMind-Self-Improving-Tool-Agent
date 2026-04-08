from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


# ============================================================
# Tool Registry - defines available tools the agent can call
# ============================================================
class ToolParameter(BaseModel):
    """Schema for a single tool parameter."""
    name: str
    type: str                               # string / number / boolean / array
    description: str
    required: bool = True
    enum: Optional[List[str]] = None        # allowed values if restricted


class ToolDefinition(BaseModel):
    """A tool available to the agent."""
    name: str
    description: str
    parameters: List[ToolParameter] = Field(default_factory=list)


# ============================================================
# Scenario - a user query with expected tool call(s)
# ============================================================
class Scenario(BaseModel):
    """A single scenario the agent must handle."""
    id: int
    user_query: str                         # what the user asked
    context: str = ""                       # optional conversation history or extra context
    available_tools: List[str]              # names of tools available for this scenario
    difficulty_tags: List[str] = Field(default_factory=list)  # e.g. ["multi_step", "refusal", "param_extraction"]
    metadata: Dict[str, str] = Field(default_factory=dict)    # extra info: domain, risk_level, etc.


# ============================================================
# Agent's Action - the tool call it decides to make
# ============================================================
class ToolCallAction(Action):
    """Action taken by the agent - one or more tool calls."""
    scenario_id: int
    tool_calls: List[Dict[str, Any]]        # [{"tool_name": "...", "parameters": {...}}, ...]
    should_refuse: bool = False             # agent can signal it should NOT call any tool
    reasoning: str = ""                     # optional chain-of-thought


# ============================================================
# Observation - what the agent sees
# ============================================================
class ToolCallObservation(Observation):
    """What the agent observes after each step."""
    scenario: Scenario                      # current scenario to handle
    tool_definitions: List[ToolDefinition]  # full schema of available tools
    queue_size: int                         # total scenarios in episode
    current_step: int                       # index of current scenario
    reward: float                           # reward from previous step
    done: bool                              # whether episode has ended


# ============================================================
# Environment State
# ============================================================
class ToolCallState(State):
    """Internal state of the environment."""
    current_index: int
    total_scenarios: int
    processed_scenario_ids: List[int]
    score: float
    done: bool
