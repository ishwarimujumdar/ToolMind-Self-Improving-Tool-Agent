"""
GRPO Training Script for Tool-Call Agent
=========================================

Designed to run on Google Colab with a free T4 GPU.
Uses Unsloth + TRL for memory-efficient GRPO training.

Usage on Colab:
  1. Create a new Colab notebook
  2. Copy each CELL below into its own cell
  3. Run cells sequentially
  4. Saves LoRA adapter + plots to Google Drive

Two training rounds:
  Round 1: GRPO on base prompts (no lessons)
  Round 2: GRPO on lesson-enriched prompts (from memory)
"""

# ============================================================
# CELL 1: Install dependencies
# ============================================================
# !pip install unsloth
# !pip install --upgrade trl>=0.14.0
# !pip install chromadb sentence-transformers datasets matplotlib

# ============================================================
# CELL 2: Clone repo & mount Drive
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')
#
# !git clone https://github.com/YOUR_USERNAME/tool-call-rl-OpenEnv.git /content/tool-call-rl-OpenEnv
# %cd /content/tool-call-rl-OpenEnv

# ============================================================
# CELL 3: Configuration
# ============================================================
import json
import os
import re
import sys
import torch
from pathlib import Path

MODEL_ID = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16

NUM_GENERATIONS = 4
MAX_COMPLETION_LENGTH = 512
LEARNING_RATE = 5e-6
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LOGGING_STEPS = 1
SAVE_STEPS = 50

SAVE_DIR = "./grpo_checkpoints"
PLOTS_DIR = "./plots"

# ============================================================
# CELL 4: Load model with Unsloth
# ============================================================
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

print(f"Model loaded: {MODEL_ID}")
model.print_trainable_parameters()

# ============================================================
# CELL 5: Load environment data (expanded 137 scenarios)
# ============================================================

sys.path.insert(0, str(Path(".").resolve()))

# Prefer expanded dataset, fallback to base
DATA_PATH = Path("data/scenarios_expanded.json")
if not DATA_PATH.exists():
    DATA_PATH = Path("data/scenarios.json")
if not DATA_PATH.exists():
    DATA_PATH = Path("/content/tool-call-rl-OpenEnv/data/scenarios_expanded.json")
if not DATA_PATH.exists():
    DATA_PATH = Path("/content/tool-call-rl-OpenEnv/data/scenarios.json")

with open(DATA_PATH) as f:
    data = json.load(f)

ALL_TOOLS = data["tools"]
TOOL_LOOKUP = {t["name"]: t for t in ALL_TOOLS}

SCENARIOS = []
LABELS = {}
for s in data["scenarios"]:
    s_copy = dict(s)
    label = s_copy.pop("label")
    SCENARIOS.append(s_copy)
    LABELS[s_copy["id"]] = label

print(f"Loaded {len(SCENARIOS)} scenarios from {DATA_PATH.name}")
print(f"Tools available: {len(ALL_TOOLS)}")


# ============================================================
# CELL 6: Define reward function (replicates env grading)
# ============================================================

def extract_json_from_text(text: str) -> dict:
    """Extract JSON from model output."""
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


def grade_completion(completion_text: str, scenario: dict, task_type: str = "hard") -> float:
    """Grade a single completion against scenario label."""
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

    # Refusal handling
    if expected_refuse:
        if should_refuse or len(tool_calls) == 0:
            return 1.0
        return 0.0

    if (should_refuse or len(tool_calls) == 0) and expected_calls:
        return 0.0

    expected_names = [tc["tool_name"] for tc in expected_calls]
    actual_names = [tc.get("tool_name", "") for tc in tool_calls]

    # Tool selection (25%)
    correct_tools = set(expected_names) & set(actual_names)
    tool_score = len(correct_tools) / max(len(expected_names), 1)
    reward += 0.25 * tool_score

    # Parameter correctness (30%)
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

    # Chain ordering (20%)
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
            order_score = 1.0 if all(valid_pos[i] < valid_pos[i+1] for i in range(len(valid_pos)-1)) else 0.2
        order_score *= len(valid_pos) / len(expected_names) if valid_pos else 0
        reward += 0.20 * order_score
    else:
        reward += 0.20 * tool_score

    # No extra calls (10%)
    extra = [n for n in actual_names if n not in expected_names]
    if not extra:
        reward += 0.10
    else:
        reward -= 0.05 * len(extra)

    # Correct count (15%)
    if len(actual_names) == len(expected_names):
        reward += 0.15
    else:
        reward -= 0.05 * abs(len(actual_names) - len(expected_names))

    # Hallucination penalty
    for name in actual_names:
        if name not in available_tools:
            reward -= 0.4

    return max(0.0, min(1.0, reward))


# ============================================================
# CELL 7: Build prompts for GRPO
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
    """Build a prompt for a scenario."""
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
    """Create HuggingFace Dataset for GRPOTrainer."""
    from datasets import Dataset

    items = []
    for scenario in scenarios:
        lessons = ""
        if lessons_fn:
            lessons = lessons_fn(scenario.get("user_query", ""))

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


# Build Round 1 dataset (no lessons)
dataset_r1 = create_dataset(SCENARIOS)
print(f"Round 1 dataset: {len(dataset_r1)} examples")
print(f"Sample prompt (truncated):\n{dataset_r1[0]['prompt'][:500]}...")


# ============================================================
# CELL 8: Define TRL reward function
# ============================================================

SCENARIO_MAP = {s["id"]: s for s in SCENARIOS}


def reward_fn(completions, scenario_id=None, **kwargs):
    """
    TRL-compatible reward function.

    Args:
        completions: list[str] — generated texts from the model
        scenario_id: list[int] — scenario IDs from the dataset (passed automatically by TRL)
    Returns:
        list[float] — reward for each completion
    """
    rewards = []
    for i, completion in enumerate(completions):
        sid = None
        if scenario_id is not None and i < len(scenario_id):
            sid = scenario_id[i]

        if sid and sid in SCENARIO_MAP:
            scenario = SCENARIO_MAP[sid]
        else:
            scenario = SCENARIOS[i % len(SCENARIOS)]

        reward = grade_completion(completion, scenario, task_type="hard")
        rewards.append(reward)

    return rewards


# Quick test
test_rewards = reward_fn(
    ['{"should_refuse": true, "reasoning": "dangerous", "tool_calls": []}'],
    scenario_id=[SCENARIOS[4]["id"]]  # refusal scenario
)
print(f"Reward function test (refusal scenario): {test_rewards}")


# ============================================================
# CELL 9: Pre-training baseline evaluation
# ============================================================

print("=" * 60)
print("BASELINE EVALUATION (before any training)")
print("=" * 60)

FastLanguageModel.for_inference(model)

baseline_rewards = []
for scenario in SCENARIOS:
    prompt = build_prompt_for_scenario(scenario)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

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
    baseline_rewards.append(reward)

    parsed = extract_json_from_text(completion)
    tool_names = [tc.get("tool_name", "") for tc in parsed.get("tool_calls", [])]
    print(f"  Scenario {scenario['id']:3d} | {str(tool_names):40s} | reward={reward:.2f}")

avg_baseline = sum(baseline_rewards) / len(baseline_rewards)
acc_baseline = sum(1 for r in baseline_rewards if r > 0.7) / len(baseline_rewards)
print(f"\nBASELINE: avg_reward={avg_baseline:.3f}, accuracy={acc_baseline:.1%}")
print("=" * 60)


# ============================================================
# CELL 10: Train with GRPO — Round 1
# ============================================================
from trl import GRPOTrainer, GRPOConfig

FastLanguageModel.for_training(model)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

training_args = GRPOConfig(
    output_dir=f"{SAVE_DIR}/round1",
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION_LENGTH,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
)

trainer_r1 = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    reward_funcs=reward_fn,
    train_dataset=dataset_r1,
)

print("Starting GRPO Round 1 training...")
train_result_r1 = trainer_r1.train()
print(f"Round 1 complete. Loss: {train_result_r1.training_loss:.4f}")

model.save_pretrained(f"{SAVE_DIR}/round1_adapter")
tokenizer.save_pretrained(f"{SAVE_DIR}/round1_adapter")
print(f"Round 1 adapter saved to {SAVE_DIR}/round1_adapter")


# ============================================================
# CELL 11: Evaluate Round 1 & collect experiences for memory
# ============================================================

FastLanguageModel.for_inference(model)

experiences_r1 = []
for scenario in SCENARIOS:
    prompt = build_prompt_for_scenario(scenario)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

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

    experiences_r1.append({
        "query": scenario["user_query"],
        "scenario_id": scenario["id"],
        "tool_sequence": tool_names,
        "reward": reward,
        "should_refuse": parsed.get("should_refuse", False),
    })

    print(f"  Scenario {scenario['id']:3d} | {str(tool_names):40s} | reward={reward:.2f}")

avg_r1 = sum(e["reward"] for e in experiences_r1) / len(experiences_r1)
acc_r1 = sum(1 for e in experiences_r1 if e["reward"] > 0.7) / len(experiences_r1)
print(f"\nROUND 1: avg_reward={avg_r1:.3f}, accuracy={acc_r1:.1%}")
print(f"Improvement over baseline: {avg_r1 - avg_baseline:+.3f}")


# ============================================================
# CELL 12: Store experiences to ChromaDB memory
# ============================================================

from memory.memory_store import MemoryStore

memory = MemoryStore(persist_dir="./data/chroma_data")

for exp in experiences_r1:
    tools_str = " -> ".join(exp["tool_sequence"]) if exp["tool_sequence"] else "REFUSED"
    outcome = "good" if exp["reward"] > 0.7 else "poor"
    lesson = f"For query like '{exp['query'][:50]}...', sequence [{tools_str}] was {outcome}"

    memory.store_experience(
        query=exp["query"],
        scenario_id=exp["scenario_id"],
        tool_sequence=exp["tool_sequence"],
        reward=exp["reward"],
        lesson=lesson,
        should_refuse=exp["should_refuse"],
        difficulty="hard",
        episode=1,
    )

print(f"Stored {memory.count()} experiences in memory")


# ============================================================
# CELL 13: Build Round 2 dataset (with lessons from memory)
# ============================================================

FastLanguageModel.for_training(model)


def get_lessons(query: str) -> str:
    return memory.format_lessons_for_prompt(query, n_results=3)


dataset_r2 = create_dataset(SCENARIOS, lessons_fn=get_lessons)
print(f"Round 2 dataset: {len(dataset_r2)} examples")
print(f"Sample enriched prompt (truncated):\n{dataset_r2[0]['prompt'][:800]}...")


# ============================================================
# CELL 14: Train with GRPO — Round 2 (with lessons)
# ============================================================

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
    save_total_limit=3,
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
)

trainer_r2 = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args_r2,
    reward_funcs=reward_fn,
    train_dataset=dataset_r2,
)

print("Starting GRPO Round 2 training (with lessons)...")
train_result_r2 = trainer_r2.train()
print(f"Round 2 complete. Loss: {train_result_r2.training_loss:.4f}")

model.save_pretrained(f"{SAVE_DIR}/round2_adapter")
tokenizer.save_pretrained(f"{SAVE_DIR}/round2_adapter")
print(f"Round 2 adapter saved to {SAVE_DIR}/round2_adapter")


# ============================================================
# CELL 15: Evaluate Round 2
# ============================================================

FastLanguageModel.for_inference(model)

experiences_r2 = []
for scenario in SCENARIOS:
    lessons = get_lessons(scenario["user_query"])
    prompt = build_prompt_for_scenario(scenario, lessons)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

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

    experiences_r2.append({"reward": reward, "tools": tool_names})
    print(f"  Scenario {scenario['id']:3d} | {str(tool_names):40s} | reward={reward:.2f}")

avg_r2 = sum(e["reward"] for e in experiences_r2) / len(experiences_r2)
acc_r2 = sum(1 for e in experiences_r2 if e["reward"] > 0.7) / len(experiences_r2)
print(f"\nROUND 2: avg_reward={avg_r2:.3f}, accuracy={acc_r2:.1%}")
print(f"Improvement over Round 1: {avg_r2 - avg_r1:+.3f}")
print(f"Improvement over baseline: {avg_r2 - avg_baseline:+.3f}")


# ============================================================
# CELL 16: Generate plots & save
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Per-scenario reward comparison
rewards_r1 = [e["reward"] for e in experiences_r1]
rewards_r2 = [e["reward"] for e in experiences_r2]
n = min(40, len(SCENARIOS))
x = range(1, n + 1)

axes[0].bar([i - 0.2 for i in x], baseline_rewards[:n], 0.25, label="Baseline", alpha=0.7, color="#ff6b6b")
axes[0].bar([i + 0.0 for i in x], rewards_r1[:n], 0.25, label="Round 1", alpha=0.7, color="#feca57")
axes[0].bar([i + 0.2 for i in x], rewards_r2[:n], 0.25, label="Round 2", alpha=0.7, color="#48dbfb")
axes[0].set_xlabel("Scenario")
axes[0].set_ylabel("Reward")
axes[0].set_title("Per-Scenario Reward (first 40)")
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 1.1)

# Plot 2: Average reward progression
stages = ["Baseline\n(untrained)", "Round 1\n(GRPO)", "Round 2\n(GRPO+Lessons)"]
avgs = [avg_baseline, avg_r1, avg_r2]
colors = ["#ff6b6b", "#feca57", "#48dbfb"]
bars = axes[1].bar(stages, avgs, color=colors, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, avgs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Average Reward")
axes[1].set_title("Training Progress")
axes[1].set_ylim(0, 1.1)

# Plot 3: Accuracy progression
accs = [acc_baseline, acc_r1, acc_r2]
bars2 = axes[2].bar(stages, [a * 100 for a in accs], color=colors, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars2, accs):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.0%}", ha="center", fontsize=11, fontweight="bold")
axes[2].set_ylabel("Accuracy (%)")
axes[2].set_title("Accuracy Progress")
axes[2].set_ylim(0, 110)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/training_results.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved to {PLOTS_DIR}/training_results.png")

# Save results as JSON for the dashboard
results_summary = {
    "baseline": {"avg_reward": avg_baseline, "accuracy": acc_baseline, "rewards": baseline_rewards},
    "round1": {"avg_reward": avg_r1, "accuracy": acc_r1, "rewards": rewards_r1},
    "round2": {"avg_reward": avg_r2, "accuracy": acc_r2, "rewards": rewards_r2},
}
with open(f"{PLOTS_DIR}/results.json", "w") as f:
    json.dump(results_summary, f, indent=2)
print(f"Results JSON saved to {PLOTS_DIR}/results.json")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Baseline:  avg_reward={avg_baseline:.3f}  accuracy={acc_baseline:.0%}")
print(f"  Round 1:   avg_reward={avg_r1:.3f}  accuracy={acc_r1:.0%}  ({avg_r1 - avg_baseline:+.3f})")
print(f"  Round 2:   avg_reward={avg_r2:.3f}  accuracy={acc_r2:.0%}  ({avg_r2 - avg_baseline:+.3f})")
print("=" * 60)


# ============================================================
# CELL 17: Push to HuggingFace Hub (optional)
# ============================================================
# Uncomment and set your HF token to push the trained adapter:
#
# from huggingface_hub import login
# login(token="your_hf_token")
#
# model.push_to_hub("your-username/tool-call-grpo-qwen3b-r2")
# tokenizer.push_to_hub("your-username/tool-call-grpo-qwen3b-r2")
# print("Model pushed to HuggingFace Hub!")
#
# To also save to Google Drive:
# !cp -r ./grpo_checkpoints /content/drive/MyDrive/grpo_checkpoints
# !cp -r ./plots /content/drive/MyDrive/grpo_plots
# print("Saved to Google Drive!")
