#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.4",
#     "transformers>=4.45",
#     "trl>=0.14.0",
#     "peft>=0.13",
#     "accelerate>=1.0",
#     "bitsandbytes>=0.43",
#     "datasets",
#     "matplotlib",
#     "huggingface_hub",
#     "chromadb",
#     "sentence-transformers",
#     "trackio",
#     "requests",
# ]
# ///
"""
GRPO Training Script for Tool-Call Agent — HuggingFace Jobs Edition
====================================================================

Self-contained UV script (PEP 723) designed to run on HuggingFace Jobs
with an A10G GPU. Trains Qwen2.5-3B-Instruct with GRPO in two rounds:

  Round 1: GRPO on base prompts (no memory)
  Round 2: GRPO on lesson-enriched prompts (from ChromaDB memory)

Usage:
  hf jobs uv run \\
      --flavor a10g-small \\
      --secrets HF_TOKEN \\
      --timeout 2h \\
      training/grpo_hf_jobs.py \\
      --output-repo your-username/tool-call-grpo-qwen3b \\
      --mode full

Modes:
  --mode fast    : 30 train / 20 eval scenarios, 1 epoch          (~25 min)
  --mode full    : 80 train / 40 eval scenarios, 2 epochs         (~75 min)
  --mode demo    : 60 train / 40 eval scenarios, 2 epochs, both rounds (~60 min)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import torch

# ============================================================
# Argument parsing (HF Jobs forwards CLI args after script path)
# ============================================================
parser = argparse.ArgumentParser(description="GRPO training on HF Jobs (A10G)")
parser.add_argument("--output-repo", type=str, required=True,
                    help="HF Hub repo to push trained adapter (e.g. user/tool-call-grpo)")
parser.add_argument("--data-url", type=str,
                    default="https://raw.githubusercontent.com/Harshitawake/tool-call-rl-OpenEnv/main/data/scenarios_expanded.json",
                    help="Raw URL to scenarios_expanded.json")
parser.add_argument("--data-fallback-url", type=str,
                    default="https://raw.githubusercontent.com/Harshitawake/tool-call-rl-OpenEnv/main/data/scenarios.json",
                    help="Fallback raw URL if expanded file not present")
parser.add_argument("--model-id", type=str,
                    default="Qwen/Qwen2.5-3B-Instruct")
parser.add_argument("--mode", type=str, choices=["fast", "full", "demo"], default="demo")
parser.add_argument("--rounds", type=int, choices=[1, 2], default=2,
                    help="Run only Round 1 (GRPO) or both rounds (GRPO + memory-enriched)")
parser.add_argument("--no-trackio", action="store_true",
                    help="Disable Trackio live monitoring")
parser.add_argument("--push-private", action="store_true",
                    help="Push trained adapter as private repo")
args = parser.parse_args()

# ============================================================
# Mode-specific hyperparameters
# ============================================================
if args.mode == "fast":
    TRAIN_SCENARIOS = 5
    EVAL_SCENARIOS = 5
    NUM_TRAIN_EPOCHS = 1
    NUM_GENERATIONS = 4
elif args.mode == "demo":
    TRAIN_SCENARIOS = 60
    EVAL_SCENARIOS = 40
    NUM_TRAIN_EPOCHS = 2
    NUM_GENERATIONS = 6
else:  # full
    TRAIN_SCENARIOS = 80
    EVAL_SCENARIOS = 40
    NUM_TRAIN_EPOCHS = 2
    NUM_GENERATIONS = 8

MODEL_ID = args.model_id
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
MAX_COMPLETION_LENGTH = 256
LEARNING_RATE = 3e-6
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 2
LOGGING_STEPS = 5
SAVE_STEPS = 100

SAVE_DIR = "./grpo_checkpoints"
PLOTS_DIR = "./plots"

print("=" * 70)
print("HF JOBS — GRPO TRAINING FOR TOOL-CALL AGENT")
print("=" * 70)
print(f"Mode:           {args.mode}")
print(f"Model:          {MODEL_ID}")
print(f"Output repo:    {args.output_repo}")
print(f"Train / Eval:   {TRAIN_SCENARIOS} / {EVAL_SCENARIOS} scenarios")
print(f"Epochs:         {NUM_TRAIN_EPOCHS}")
print(f"Generations:    {NUM_GENERATIONS}")
print(f"Rounds:         {args.rounds}")
print(f"Trackio:        {'disabled' if args.no_trackio else 'enabled'}")
print("=" * 70)

# ============================================================
# Trackio setup (optional — graceful fallback)
# ============================================================
USE_TRACKIO = not args.no_trackio
if USE_TRACKIO:
    try:
        import trackio
        trackio.init(
            project="tool-call-grpo",
            name=f"hfjobs-{args.mode}-{int(time.time())}",
            config={
                "model": MODEL_ID,
                "mode": args.mode,
                "train_scenarios": TRAIN_SCENARIOS,
                "epochs": NUM_TRAIN_EPOCHS,
                "num_generations": NUM_GENERATIONS,
                "learning_rate": LEARNING_RATE,
                "rounds": args.rounds,
            },
        )
        print(f"Trackio initialized — view live at: {trackio.get_url() if hasattr(trackio, 'get_url') else 'https://huggingface.co/trackio'}")
    except Exception as e:
        print(f"WARNING: Trackio init failed ({e}). Continuing without it.")
        USE_TRACKIO = False

# ============================================================
# Fetch scenarios from GitHub
# ============================================================
def fetch_scenarios(url: str, fallback: str) -> dict:
    for u in [url, fallback]:
        try:
            print(f"Fetching scenarios from: {u}")
            r = requests.get(u, timeout=30)
            r.raise_for_status()
            data = r.json()
            print(f"  Loaded {len(data['scenarios'])} scenarios, {len(data['tools'])} tools")
            return data
        except Exception as e:
            print(f"  Failed: {e}")
    raise RuntimeError("Could not fetch scenarios from any URL")

raw_data = fetch_scenarios(args.data_url, args.data_fallback_url)
ALL_TOOLS = raw_data["tools"]
TOOL_LOOKUP = {t["name"]: t for t in ALL_TOOLS}

SCENARIOS = []
LABELS = {}
for s in raw_data["scenarios"]:
    s_copy = dict(s)
    label = s_copy.pop("label")
    SCENARIOS.append(s_copy)
    LABELS[s_copy["id"]] = label

SCENARIO_MAP = {s["id"]: s for s in SCENARIOS}
print(f"Total scenarios available: {len(SCENARIOS)}")

# ============================================================
# Load model with vanilla transformers + peft + bitsandbytes (QLoRA)
# (avoids Unsloth's GRPO+LoRA dtype edge cases on A10G)
# ============================================================
print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

VANILLA_MODEL_ID = MODEL_ID if not MODEL_ID.startswith("unsloth/") else MODEL_ID.replace(
    "unsloth/", "Qwen/"
).replace("-bnb-4bit", "")

print(f"Loading {VANILLA_MODEL_ID} with 4-bit QLoRA + bf16 compute...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    VANILLA_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(VANILLA_MODEL_ID)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA * 2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print(f"Model loaded: {VANILLA_MODEL_ID}")
model.print_trainable_parameters()

# ============================================================
# Reward function (same logic as local grpo_train.py)
# ============================================================
def extract_json_from_text(text: str) -> dict:
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


def grade_completion(completion_text: str, scenario: dict) -> float:
    label = LABELS.get(scenario["id"])
    if label is None:
        return 0.0

    parsed = extract_json_from_text(completion_text)
    tool_calls = parsed.get("tool_calls", [])
    should_refuse = parsed.get("should_refuse", False)

    expected_calls = label["expected_tool_calls"]
    expected_refuse = label["should_refuse"]
    required_params = label.get("required_params", {})
    chain_order_matters = label.get("chain_order_matters", False)
    available_tools = scenario.get("available_tools", [])

    reward = 0.0

    if expected_refuse:
        if should_refuse or len(tool_calls) == 0:
            return 1.0
        return 0.0

    if (should_refuse or len(tool_calls) == 0) and expected_calls:
        return 0.0

    expected_names = [tc["tool_name"] for tc in expected_calls]
    actual_names = [tc.get("tool_name", "") for tc in tool_calls]

    correct_tools = set(expected_names) & set(actual_names)
    tool_score = len(correct_tools) / max(len(expected_names), 1)
    reward += 0.25 * tool_score

    param_scores = []
    for exp_call in expected_calls:
        exp_name = exp_call["tool_name"]
        matching = [tc for tc in tool_calls if tc.get("tool_name") == exp_name]
        if matching:
            params = matching[0].get("parameters", {})
            exp_params = exp_call.get("parameters", {})
            req = required_params.get(exp_name, [])
            total = max(len(req), 1)
            ps = 0.0
            for pname in req:
                if pname in params:
                    ps += 0.5
                    exp_val = exp_params.get(pname)
                    act_val = params.get(pname)
                    if exp_val is not None and not str(exp_val).startswith("<"):
                        a = str(act_val).strip().lower() if act_val else ""
                        e = str(exp_val).strip().lower()
                        if a == e or (len(e) > 10 and (e in a or a in e)):
                            ps += 0.5
            param_scores.append(ps / total)
        else:
            param_scores.append(0.0)

    if param_scores:
        reward += 0.30 * (sum(param_scores) / len(param_scores))

    if chain_order_matters and len(expected_calls) > 1:
        positions = []
        for exp_name in expected_names:
            for i, act in enumerate(actual_names):
                if act == exp_name:
                    positions.append(i)
                    break
            else:
                positions.append(-1)
        valid_pos = [p for p in positions if p >= 0]
        if len(valid_pos) <= 1:
            order_score = 1.0 if valid_pos else 0.0
        else:
            order_score = 1.0 if all(valid_pos[i] < valid_pos[i + 1] for i in range(len(valid_pos) - 1)) else 0.2
        order_score *= len(valid_pos) / len(expected_names) if valid_pos else 0
        reward += 0.20 * order_score
    else:
        reward += 0.20 * tool_score

    extra = [n for n in actual_names if n not in expected_names]
    if not extra:
        reward += 0.10
    else:
        reward -= 0.05 * len(extra)

    if len(actual_names) == len(expected_names):
        reward += 0.15
    else:
        reward -= 0.05 * abs(len(actual_names) - len(expected_names))

    for name in actual_names:
        if name not in available_tools:
            reward -= 0.4

    return max(0.0, min(1.0, reward))


# ============================================================
# Prompt builder
# ============================================================
SYSTEM_PROMPT = """You are an AI agent that decides which tools to call for a given user query.

RULES:
1. Pick the correct tool(s) and provide the correct parameters.
2. For multi-step tasks, list tool calls in the correct execution order.
3. If the query is dangerous (data deletion, injection), REFUSE.
4. If no tool is needed, REFUSE.
5. Never call tools not in the available list.

Respond ONLY in JSON:
{"should_refuse": false, "reasoning": "...", "tool_calls": [{"tool_name": "...", "parameters": {...}}]}
If refusing:
{"should_refuse": true, "reasoning": "...", "tool_calls": []}"""


def build_prompt_for_scenario(scenario: dict, lessons_text: str = "") -> str:
    available = scenario.get("available_tools", [])
    tool_defs = [TOOL_LOOKUP[t] for t in available if t in TOOL_LOOKUP]

    tools_block = ""
    for td in tool_defs:
        params = ", ".join(
            f"{p['name']}: {p['type']}{' (REQUIRED)' if p.get('required', True) else ''}"
            for p in td.get("parameters", [])
        )
        tools_block += f"  {td['name']}: {td['description']} [{params}]\n"

    query = scenario.get("user_query", "")
    context = scenario.get("context", "")
    ctx_str = f"\nContext: {context}" if context else ""

    prompt = f"USER QUERY: {query}{ctx_str}\n\nAVAILABLE TOOLS:\n{tools_block}"
    if lessons_text:
        prompt += f"\n{lessons_text}\n\nUse these lessons to guide your decision."
    prompt += "\n\nDecide which tool(s) to call (or refuse). Respond in JSON."
    return prompt


def create_dataset(scenarios: list, lessons_fn=None):
    from datasets import Dataset
    items = []
    for scenario in scenarios:
        lessons = lessons_fn(scenario.get("user_query", "")) if lessons_fn else ""
        prompt = build_prompt_for_scenario(scenario, lessons)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        items.append({
            "prompt": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
            "scenario_id": scenario["id"],
        })
    return Dataset.from_list(items)


# ============================================================
# TRL reward callback
# ============================================================
def reward_fn(completions, scenario_id=None, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        sid = None
        if scenario_id is not None and i < len(scenario_id):
            sid = scenario_id[i]
        scenario = SCENARIO_MAP.get(sid) if sid else SCENARIOS[i % len(SCENARIOS)]
        rewards.append(grade_completion(completion, scenario))
    if USE_TRACKIO:
        try:
            trackio.log({
                "reward/mean": sum(rewards) / len(rewards),
                "reward/min": min(rewards),
                "reward/max": max(rewards),
            })
        except Exception:
            pass
    return rewards


# ============================================================
# Evaluation helper
# ============================================================
def evaluate_model(scenarios, lessons_fn=None, label="EVAL"):
    model.eval()
    rewards = []
    experiences = []
    print(f"\n{'-' * 70}\n{label} ({len(scenarios)} scenarios)\n{'-' * 70}")
    for scenario in scenarios:
        lessons = lessons_fn(scenario["user_query"]) if lessons_fn else ""
        prompt = build_prompt_for_scenario(scenario, lessons)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1536).to(model.device)

        completion = ""
        tool_names = []
        parsed = {}
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_COMPLETION_LENGTH,
                    temperature=0.3,
                    do_sample=True,
                )
            completion = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )
            reward = grade_completion(completion, scenario)
            parsed = extract_json_from_text(completion)
            tool_names = [tc.get("tool_name", "") for tc in parsed.get("tool_calls", [])]
        except Exception as e:
            print(f"  Scenario {scenario['id']:3d} | ERROR: {str(e)[:60]}")
            reward = 0.0

        rewards.append(reward)
        experiences.append({
            "query": scenario["user_query"],
            "scenario_id": scenario["id"],
            "tool_sequence": tool_names,
            "reward": reward,
            "should_refuse": parsed.get("should_refuse", False),
        })
        print(f"  Scenario {scenario['id']:3d} | {str(tool_names):40s} | reward={reward:.2f}")

    avg = sum(rewards) / len(rewards)
    acc = sum(1 for r in rewards if r > 0.7) / len(rewards)
    print(f"\n{label}: avg_reward={avg:.3f}, accuracy={acc:.1%}")
    if USE_TRACKIO:
        try:
            trackio.log({f"eval/{label}/avg_reward": avg, f"eval/{label}/accuracy": acc})
        except Exception:
            pass
    return rewards, experiences, avg, acc


# ============================================================
# Build datasets
# ============================================================
dataset_r1 = create_dataset(SCENARIOS[:TRAIN_SCENARIOS])
print(f"\nRound 1 dataset: {len(dataset_r1)} examples")

# ============================================================
# Pre-training baseline evaluation
# ============================================================
print("\n" + "=" * 70)
print("BASELINE EVALUATION (before training)")
print("=" * 70)
baseline_rewards, _, avg_baseline, acc_baseline = evaluate_model(
    SCENARIOS[:EVAL_SCENARIOS], label="BASELINE"
)

# ============================================================
# Round 1: GRPO training
# ============================================================
print("\n" + "=" * 70)
print("ROUND 1: GRPO TRAINING (no memory)")
print("=" * 70)
from trl import GRPOTrainer, GRPOConfig

model.train()
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

training_args_r1 = GRPOConfig(
    output_dir=f"{SAVE_DIR}/round1",
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION_LENGTH,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    bf16=True,
    report_to=("trackio" if USE_TRACKIO else "none"),
    remove_unused_columns=False,
)

trainer_r1 = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args_r1,
    reward_funcs=reward_fn,
    train_dataset=dataset_r1,
)
print(f"Training Round 1 on {len(dataset_r1)} scenarios...")
train_result_r1 = trainer_r1.train()
print(f"Round 1 complete. Loss: {train_result_r1.training_loss:.4f}")

model.save_pretrained(f"{SAVE_DIR}/round1_adapter")
tokenizer.save_pretrained(f"{SAVE_DIR}/round1_adapter")

print("\n" + "=" * 70)
print("ROUND 1 EVALUATION")
print("=" * 70)
rewards_r1, experiences_r1, avg_r1, acc_r1 = evaluate_model(
    SCENARIOS[:EVAL_SCENARIOS], label="ROUND 1"
)
print(f"Improvement over baseline: {avg_r1 - avg_baseline:+.3f}")

# ============================================================
# Round 2: GRPO with memory-enriched prompts
# ============================================================
avg_r2 = avg_r1
acc_r2 = acc_r1
rewards_r2 = rewards_r1

if args.rounds == 2:
    print("\n" + "=" * 70)
    print("BUILDING MEMORY FROM ROUND 1 EXPERIENCES")
    print("=" * 70)

    import chromadb
    from chromadb.config import Settings
    chroma_client = chromadb.PersistentClient(
        path="./data/chroma_data",
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(
        name="tool_experiences",
        metadata={"hnsw:space": "cosine"},
    )

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def store_experience(query, sid, tools, reward, should_refuse):
        tools_str = " -> ".join(tools) if tools else "REFUSED"
        outcome = "good" if reward > 0.7 else "poor"
        lesson = f"For query like '{query[:60]}', sequence [{tools_str}] was {outcome} (reward={reward:.2f})"
        emb = embedder.encode([query])[0].tolist()
        collection.add(
            ids=[f"r1_s{sid}"],
            embeddings=[emb],
            documents=[lesson],
            metadatas=[{
                "query": query,
                "scenario_id": sid,
                "reward": float(reward),
                "should_refuse": bool(should_refuse),
                "tools": tools_str,
            }],
        )

    for exp in experiences_r1:
        try:
            store_experience(
                exp["query"], exp["scenario_id"],
                exp["tool_sequence"], exp["reward"], exp["should_refuse"],
            )
        except Exception as e:
            print(f"  Skip exp {exp['scenario_id']}: {e}")

    print(f"Stored {collection.count()} experiences in ChromaDB memory")

    def get_lessons(query: str) -> str:
        try:
            emb = embedder.encode([query])[0].tolist()
            res = collection.query(query_embeddings=[emb], n_results=3)
            docs = res.get("documents", [[]])[0]
            if not docs:
                return ""
            return "PAST LESSONS:\n" + "\n".join(f"- {d}" for d in docs)
        except Exception:
            return ""

    print("\n" + "=" * 70)
    print("ROUND 2: GRPO TRAINING (with memory lessons)")
    print("=" * 70)
    model.train()

    dataset_r2 = create_dataset(SCENARIOS[:TRAIN_SCENARIOS], lessons_fn=get_lessons)
    print(f"Round 2 dataset: {len(dataset_r2)} examples (with memory lessons)")

    training_args_r2 = GRPOConfig(
        output_dir=f"{SAVE_DIR}/round2",
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        learning_rate=LEARNING_RATE * 0.5,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=True,
        report_to=("trackio" if USE_TRACKIO else "none"),
        remove_unused_columns=False,
    )

    trainer_r2 = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args_r2,
        reward_funcs=reward_fn,
        train_dataset=dataset_r2,
    )
    print(f"Training Round 2 on {len(dataset_r2)} scenarios...")
    train_result_r2 = trainer_r2.train()
    print(f"Round 2 complete. Loss: {train_result_r2.training_loss:.4f}")

    model.save_pretrained(f"{SAVE_DIR}/round2_adapter")
    tokenizer.save_pretrained(f"{SAVE_DIR}/round2_adapter")

    print("\n" + "=" * 70)
    print("ROUND 2 EVALUATION")
    print("=" * 70)
    rewards_r2, experiences_r2, avg_r2, acc_r2 = evaluate_model(
        SCENARIOS[:EVAL_SCENARIOS], lessons_fn=get_lessons, label="ROUND 2"
    )
    print(f"Improvement over Round 1: {avg_r2 - avg_r1:+.3f}")
    print(f"Improvement over baseline: {avg_r2 - avg_baseline:+.3f}")

# ============================================================
# Plots & summary
# ============================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
n = min(EVAL_SCENARIOS, len(baseline_rewards))
x = list(range(1, n + 1))

axes[0].bar([i - 0.25 for i in x], baseline_rewards[:n], 0.25, label="Baseline", alpha=0.8, color="#ff6b6b")
axes[0].bar([i + 0.00 for i in x], rewards_r1[:n], 0.25, label="Round 1", alpha=0.8, color="#feca57")
if args.rounds == 2:
    axes[0].bar([i + 0.25 for i in x], rewards_r2[:n], 0.25, label="Round 2", alpha=0.8, color="#48dbfb")
axes[0].set_xlabel("Scenario")
axes[0].set_ylabel("Reward")
axes[0].set_title(f"Per-Scenario Reward (first {n})")
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, 1.1)

stages = ["Baseline", "Round 1\n(GRPO)"]
avgs = [avg_baseline, avg_r1]
accs = [acc_baseline, acc_r1]
colors = ["#ff6b6b", "#feca57"]
if args.rounds == 2:
    stages.append("Round 2\n(GRPO+Memory)")
    avgs.append(avg_r2)
    accs.append(acc_r2)
    colors.append("#48dbfb")

bars = axes[1].bar(stages, avgs, color=colors, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, avgs):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Average Reward")
axes[1].set_title("Training Progress")
axes[1].set_ylim(0, 1.1)

bars2 = axes[2].bar(stages, [a * 100 for a in accs], color=colors, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars2, accs):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0%}", ha="center", fontsize=11, fontweight="bold")
axes[2].set_ylabel("Accuracy (%)")
axes[2].set_title("Accuracy Progress")
axes[2].set_ylim(0, 110)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/training_results.png", dpi=150, bbox_inches="tight")
print(f"Plot saved to {PLOTS_DIR}/training_results.png")

results_summary = {
    "config": {
        "mode": args.mode,
        "model": MODEL_ID,
        "train_scenarios": TRAIN_SCENARIOS,
        "eval_scenarios": EVAL_SCENARIOS,
        "epochs": NUM_TRAIN_EPOCHS,
        "num_generations": NUM_GENERATIONS,
        "rounds": args.rounds,
    },
    "baseline": {"avg_reward": avg_baseline, "accuracy": acc_baseline, "rewards": baseline_rewards},
    "round1": {"avg_reward": avg_r1, "accuracy": acc_r1, "rewards": rewards_r1},
}
if args.rounds == 2:
    results_summary["round2"] = {"avg_reward": avg_r2, "accuracy": acc_r2, "rewards": rewards_r2}

with open(f"{PLOTS_DIR}/results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"  Baseline:  avg_reward={avg_baseline:.3f}  accuracy={acc_baseline:.0%}")
print(f"  Round 1:   avg_reward={avg_r1:.3f}  accuracy={acc_r1:.0%}  ({avg_r1 - avg_baseline:+.3f})")
if args.rounds == 2:
    print(f"  Round 2:   avg_reward={avg_r2:.3f}  accuracy={acc_r2:.0%}  ({avg_r2 - avg_baseline:+.3f})")
print("=" * 70)

# ============================================================
# Push to HuggingFace Hub
# ============================================================
print("\n" + "=" * 70)
print(f"PUSHING ADAPTER TO HUB: {args.output_repo}")
print("=" * 70)
try:
    final_adapter_dir = f"{SAVE_DIR}/round2_adapter" if args.rounds == 2 else f"{SAVE_DIR}/round1_adapter"

    model.push_to_hub(args.output_repo, private=args.push_private)
    tokenizer.push_to_hub(args.output_repo, private=args.push_private)
    print(f"Adapter pushed to: https://huggingface.co/{args.output_repo}")

    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=f"{PLOTS_DIR}/training_results.png",
            path_in_repo="training_results.png",
            repo_id=args.output_repo,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj=f"{PLOTS_DIR}/results.json",
            path_in_repo="results.json",
            repo_id=args.output_repo,
            repo_type="model",
        )
        print("Plots & results.json uploaded to model repo")
    except Exception as e:
        print(f"WARNING: Could not upload artifacts: {e}")

except Exception as e:
    print(f"ERROR: Push to hub failed: {e}")
    print("Trained adapter is still saved locally in the job container.")

if USE_TRACKIO:
    try:
        trackio.finish()
    except Exception:
        pass

print("\nDone.")
