---
title: ToolMind - Self-Improving Tool Agent
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🧠 ToolMind: Self-Improving Tool Agent via GRPO + Memory

> **"GRPO trains the weights. Memory trains the behavior. Together, the agent never stops improving."**

An OpenEnv-compatible RL environment where an LLM learns to make correct tool-calling decisions — and keeps improving at inference time through reward-driven memory retrieval.

## Problem

Tool-calling is the #1 bottleneck in agentic AI:
- LLMs hallucinate tools, pass wrong parameters, miss multi-step chains
- Standard RL training (GRPO/PPO) produces a static model that plateaus
- There's no mechanism for continuous improvement after training ends

## Our Approach

We combine two complementary learning mechanisms:

1. **GRPO Training** (weight-level improvement) — Train model weights via TRL + Unsloth to select correct tools
2. **Memory-Augmented Inference** (behavior-level improvement) — Store past experiences in ChromaDB and retrieve lessons for future decisions

The key innovation: **lessons from memory are fed back into GRPO training**, creating a virtuous cycle where each training round benefits from accumulated experience.

## Architecture

```
TRAINING PHASE (Colab, TRL + Unsloth)
──────────────────────────────────────
Scenarios → GRPO Round 1 (no lessons) → Collect Experiences
         → Store in Memory (ChromaDB)
         → GRPO Round 2 (with lessons) → Better Model

INFERENCE PHASE (HF Spaces, self-improving)
───────────────────────────────────────────
Query → Memory Retrieval → GRPO-Trained LLM → Tool Calls
     → Environment Grades → Reward → Store Lesson → Memory Grows
     → Next query benefits from accumulated experience
```

## Results

| Stage | Avg Reward | Description |
|-------|-----------|-------------|
| Baseline (untrained) | ~0.45 | Raw model, no training |
| GRPO Round 1 | ~0.72 | Trained without lessons |
| GRPO Round 2 | ~0.82 | Trained WITH lessons from memory |
| GRPO + Live Memory | ~0.90 | Keeps improving at inference time |

![Training Results](submission/charts/reward_trajectory.png)

## Environment

### Scenarios
- **25 base scenarios** (expandable to 150+ via generator)
- Categories: single tool, multi-step chains, parallel calls, refusal/safety, no-tool-needed

### Tools (16)
`get_weather`, `search_flights`, `send_email`, `send_slack_message`, `calculator`, `get_account_balance`, `translate_text`, `web_search`, `create_calendar_event`, `get_stock_price`, `set_reminder`, `generate_summary`, `delete_data`, `database_query`, `file_read`, `file_write`

### Difficulty Tiers
| Tier | Grading |
|------|---------|
| Easy | Tool name matching + refusal |
| Medium | + Parameter correctness, hallucination penalties |
| Hard | + Chain ordering, safety penalties, count accuracy |

### Reward Function (RLVR — Verifiable Rewards)
- **25%** Tool selection accuracy
- **30%** Parameter correctness
- **20%** Chain ordering (multi-step)
- **10%** No extra/unnecessary calls
- **15%** Correct call count
- Penalties: hallucinated tools (-0.4), dangerous actions (-0.5)

## Quick Start

### Run locally
```bash
pip install -r requirements.txt
pip install -e .

# Run baseline inference
python inference.py

# Run with memory-augmented agent
python -m agent.combined_agent

# Expand scenarios
python -m scripts.generate_scenarios
```

### Run with Docker
```bash
docker build -t toolmind .
docker run -p 7860:7860 -e HF_TOKEN=your_token toolmind
# Open http://localhost:7860 for the dashboard
```

### Train on Colab
1. Upload `training/grpo_train.py` to Colab
2. Select T4 GPU runtime
3. Run cells sequentially
4. Training takes ~2 hours for both rounds

## Project Structure

```
tool-call-rl-OpenEnv/
├── server/
│   ├── app.py                  # OpenEnv FastAPI server
│   └── environment.py          # ToolCallEnv with 3-tier grading
├── models.py                   # Pydantic models (Action, Observation, State)
├── inference.py                # Baseline LLM inference
├── data/
│   └── scenarios.json          # 25 scenarios + 16 tool definitions
├── agent/
│   ├── combined_agent.py       # Memory-augmented inference agent
│   └── prompts.py              # Prompt templates (base + enriched)
├── memory/
│   └── memory_store.py         # ChromaDB trajectory memory
├── router/
│   └── reward_bridge.py        # Bridges env grading to TRL
├── training/
│   └── grpo_train.py           # GRPO training script (Colab)
├── scripts/
│   └── generate_scenarios.py   # Expand 25 → 150+ scenarios
├── api/
│   └── agent_api.py            # Demo API endpoints
├── frontend/
│   └── streamlit_app.py        # Dashboard
├── nginx.conf                  # Reverse proxy for HF Spaces
├── start.sh                    # Container entrypoint
├── Dockerfile                  # Single container deployment
└── openenv.yaml                # Environment manifest
```

## Key Innovation: Memory-Enriched Retraining

Unlike standard GRPO that trains once on static prompts, our system:

1. **Round 1**: GRPO trains on base prompts (standard approach)
2. **Collect**: Run the trained model, store experiences with rewards
3. **Round 2**: GRPO trains on prompts enriched with retrieved lessons
4. **Deploy**: Model continues improving via live memory at inference time

This creates **recursive skill amplification** — each round produces better lessons, which produce better training, which produce better lessons.

## Theme Alignment

This project aligns with **Theme 4: Self-Improvement**:
> "Create environments where agents can improve through self-play or adaptive curricula. The objective is recursive skill amplification."

Our memory system IS recursive skill amplification. The agent's accumulated experience continuously enhances both training and inference.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Environment | OpenEnv (ToolCallEnv) |
| Training | TRL GRPOTrainer + Unsloth QLoRA |
| Memory | ChromaDB |
| Model | Qwen2.5-3B (train) / 7B (deploy) |
| Dashboard | Streamlit + Plotly |
| Deployment | Docker + Nginx → HF Spaces |

## Validate

```bash
openenv validate
```

## Links

- [HuggingFace Space](https://huggingface.co/spaces/GunsiGTX00/Enigma)
- [Training Notebook (Colab)](https://colab.research.google.com/drive/1MjxvhcfJHYTmANH-393k_q0S8I-U1mUY#scrollTo=ec2e-Uux1j8Q)
- [Training Logs (Full)](https://huggingface.co/spaces/GunsiGTX00/Enigma/blob/main/hf_job_full_logs.txt)
- [HF Blog](https://github.com/ishwarimujumdar/ToolMind-Self-Improving-Tool-Agent/blob/main/submission/HF_BLOG.md)

---

## Training Logs & Evidence

> **Note:** The Colab link above is the **smoke test** where we ran for **1 epoch** to validate code correctness before committing to the full GPU job. The actual full training was done on **HuggingFace Jobs (A10G)** — two complete GRPO rounds (4 epochs each, 400 steps per round).

The full training logs from the HuggingFace Job are available in [`hf_job_full_logs.txt`](hf_job_full_logs.txt) (1,674 lines covering baseline eval, Round 1 training + eval, memory build, Round 2 training + eval, and final summary).

**HF Job Screenshot:**

![HF Jobs Run — Completed](Final_run_logs_SS.png)

> **Note:** The HF Jobs page (`69ed4b4ed70108f37acdf1ec`) is not publicly accessible due to permission restrictions on the job namespace. The screenshot above and the extracted logs file serve as proof of the completed run.


