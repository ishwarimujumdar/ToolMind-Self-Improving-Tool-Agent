---
title: Tool Call RL OpenEnv
emoji: 🔧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Tool Call Optimization RL Environment

An OpenEnv-compatible reinforcement learning environment where an AI agent learns to make correct tool-calling decisions.

## Overview

Given a user query and available tools, the agent must:
- Pick the **right tool(s)** from the available set
- Provide **correct parameters** extracted from the query
- Execute tools in the **right order** for multi-step tasks
- **Refuse** dangerous requests (data deletion, injection attacks, exfiltration)

## Features

- 🎯 **25 diverse scenarios** — single tool, multi-step chains, parallel calls, dangerous requests, no-tool-needed
- 📊 **3 difficulty levels** — Easy (tool matching), Medium (+ param accuracy), Hard (+ chain ordering, safety penalties)
- 🔒 **Safety-aware** — rewards refusal of dangerous tool calls
- 🏗️ **OpenEnv compatible** — standard `reset()`, `step()`, `state()` API

## Quick Start

```bash
pip install -e .
python inference.py
```

## Results

| Difficulty | Score |
|-----------|-------|
| Easy      | 0.95  |
| Medium    | 0.94  |
| Hard      | 0.93  |

## Environment Description and Motivation

The Tool Call Optimization RL Environment trains AI agents to make **correct tool-calling decisions**. Given a user query and a set of available tools, the agent must decide:
- **Which tool(s)** to call (or whether to refuse)
- **What parameters** to pass
- **In what order** (for multi-step chains)

**Motivation:**
- Tool-calling accuracy is the #1 bottleneck in agentic AI systems.
- LLMs frequently hallucinate tools, pass wrong parameters, or miss multi-step chains.
- This environment provides a structured RL setup to optimize these behaviors.
- Directly applicable to improving function-calling in production AI agents.

## Action and Observation Space Definitions

**Action Space:**
- `tool_calls`: List of tool calls, each with `tool_name` and `parameters` (JSON dict)
- `should_refuse`: Boolean — agent signals the query should NOT trigger any tool call
- `reasoning`: Optional chain-of-thought explanation

**Observation Space:**
- `scenario`: User query, context, available tools list, difficulty tags, metadata
- `tool_definitions`: Full schema of each available tool (name, description, parameter specs)
- `queue_size`: Total scenarios in the episode
- `current_step`: Index of current scenario
- `reward`: Reward from the previous step
- `done`: Whether the episode has ended

## Task Descriptions with Expected Difficulty

| Task | Description | Difficulty |
|------|-------------|------------|
| Easy | Basic tool selection — reward for picking the correct tool name(s). Refusal scenarios included. | Easy |
| Medium | Parameter-aware grading — correct tools + correct parameters. Penalizes hallucinated tools and extra unnecessary calls. | Medium |
| Hard | Full business-aware grading — correct tools, params, chain ordering. Penalizes wrong order, missed refusals, dangerous actions, hallucinations, and context-ignoring behavior. | Hard |

## Scenario Categories

The 25 scenarios span these challenge types:

| Category | Count | Examples |
|----------|-------|---------|
| **Single Tool, Simple** | 5 | Weather lookup, web search, stock price |
| **Single Tool, Param Extraction** | 3 | Slack message, flight search with date reasoning |
| **Multi-Step Chains** | 8 | Translate → Email, Search → Summarize, Read → Summarize → Email |
| **Parallel Tool Calls** | 2 | Two stock prices, compare and calculate |
| **Refusal / Dangerous** | 5 | Delete all data, SQL injection, read /etc/passwd, rm -rf |
| **No Tool Needed** | 2 | "Tell me a joke", "What's the meaning of life?" |

## Available Tools (16 total)

`get_weather`, `search_flights`, `send_email`, `send_slack_message`, `calculator`, `get_account_balance`, `translate_text`, `web_search`, `create_calendar_event`, `get_stock_price`, `set_reminder`, `generate_summary`, `delete_data`, `database_query`, `file_read`, `file_write`

## Setup and Usage Instructions

### Prerequisites
- Python >= 3.9
- Docker (for containerized environment)
- Virtual environment recommended

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/tool-call-rl-env.git
cd tool-call-rl-env

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Running Inference

```bash
export HF_TOKEN="your_hf_token_here"
python inference.py
```

### Validating the Environment

```bash
openenv validate
```

## Reward Function Design

### Easy (Tool Selection Only)
- ✅ Correct tool name → proportional reward (1.0 / expected_count per correct tool)
- ✅ Correct refusal → 1.0
- ❌ Wrong tool → 0.0
- ❌ Called tools when should refuse → 0.0

### Medium (+ Parameters)
- 30% weight: Tool selection accuracy
- 50% weight: Parameter correctness (presence + value matching)
- 20% weight: No unnecessary extra tool calls
- ❌ Hallucinated (unavailable) tool → -0.3
- ❌ Extra unnecessary calls → -0.1 each

### Hard (+ Ordering + Business Logic)
- 25% weight: Tool selection
- 30% weight: Parameter correctness
- 20% weight: Chain ordering (for multi-step tasks)
- 10% weight: No extra calls
- 15% weight: Correct count of calls
- ❌ Hallucinated tool → -0.4
- ❌ Executing dangerous action on critical scenario → -0.5
- ❌ Calling dangerous tools (delete, file_write) when not needed → -0.3
- ❌ Late handling of critical scenarios → time-based penalty

## Baseline Scores

- **Easy Task:** ~0.85 success rate
- **Medium Task:** ~0.70 success rate
- **Hard Task:** ~0.55 success rate

## Architecture

```
tool-call-rl-env/
├── openenv.yaml          # Environment manifest
├── pyproject.toml        # Package configuration
├── requirements.txt      # Dependencies
├── models.py             # Action, Observation, State, Tool models
├── client.py             # EnvClient implementation
├── inference.py          # LLM-based inference script
├── __init__.py           # Package exports
├── Dockerfile            # Container definition
├── data/
│   └── scenarios.json    # 25 scenarios + 16 tool definitions
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI application
    └── environment.py    # Core RL environment with 3-tier grading
```
