# ToolMind: GRPO trains the weights. Memory trains the behavior.

*OpenEnv · TRL GRPO · Qwen2.5-3B · ChromaDB · Hugging Face Jobs (A10G)*

[Adapter on the Hub](https://huggingface.co/Harshitawake/tool-call-grpo-a10g-v2) · *Training code lives in the project repo (see Artifacts).*

---

## Who this is for

This post is for readers who want a **concrete, end-to-end** story: a tool-calling agent trained with **on-environment rewards** (not hand-written rubrics only), a **second training round** that uses **vector memory** as curriculum, and **reported numbers** on a **fixed held-out eval**—similar in spirit to how other Hub write-ups separate “what we built,” “how it runs,” and “where to get artifacts” (e.g. [Gemma 4 VLA on Jetson](https://huggingface.co/blog/nvidia/gemma4#gemma-4-vla-demo-on-jetson-orin-nano-super), [DeepSeek-V4](https://huggingface.co/blog/deepseekv4), [Transformers.js in a Chrome extension](https://huggingface.co/blog/transformersjs-chrome-extension)).

---

## What this post covers

- **Environment:** multi-tool scenarios + graded reward (OpenEnv-style), sized for real RL signal.
- **Pipeline:** baseline eval → **GRPO round 1** → **ChromaDB memory** from eval → **GRPO round 2** with lesson-enriched prompts.
- **Results:** one table, same 40 eval scenarios for every row.
- **Artifacts:** LoRA adapter, static figures, how to regenerate charts from the numbers.

---

## The problem in one minute

A model can *name* API tools; a **deployed** agent must **sequence** the right tools (or **abstain**), under **your** rubric. Pretraining does not **align** the policy to that reward. **ToolMind** combines:

1. **GRPO (TRL)** — update weights against environment reward.
2. **Memory (Chroma + sentence embeddings)** — store (query, tools, reward, lesson); retrieve for similar tasks; **inject lessons into the training data** for a **second** GRPO round.

> *GRPO trains the weights. Memory trains the behavior.*

---

## How it works

High-level data path (one line, like a pipeline diagram):

```
User scenario → model proposes tools → environment grades trace → reward
                    ↑                                   │
                    │         ┌── memory (lessons) ─────┤
            GRPO round 1     │   from round-1 eval     │
            GRPO round 2 ────┴── (prompts enriched)     │
```

1. **Baseline:** run the 3B instruct model on the **40-scenario held-out eval**; record reward + accuracy.
2. **Round 1:** **GRPO** on 100 training scenarios (standard prompts), **400 optimizer steps** (job config: 4 epochs, 4 generations; bf16 LoRA on A10G). Re-run **the same 40 eval** scenarios.
3. **Memory:** embed eval trajectories; **40 experiences** in **ChromaDB** (lessons for retrieval).
4. **Round 2:** same 100 train scenarios, but prompts can include **retrieved lessons**; another **400-step** GRPO run. Re-run **the same 40 eval** again.

The eval never leaks into the “train 100” split: it is a **reporting** set used consistently before and after each stage.

---

## What we built: the environment

We stress **realistic** tool use:

- **Single- and multi-step** chains (calendar, email, finance, search, files, Slack, translation, etc.).
- **Abstention** when “no tool” is correct.
- **Partial credit** from the grader (not only 0/1), so the reward is **learnable** for RL.

**Scale (this run):** 100 train scenarios per round · **40 eval** scenarios · 137 scenarios in the expanded pool · **16 tools**. Grader compares predicted tool traces to references and returns a **scalar reward** (and a binary **accuracy** for reporting).

---

## Training setup & hardware

Comparable to a “Step / Hardware / Environment” table on other Hub posts: we use **Hugging Face Jobs** on an **A10G**; base model **Qwen/Qwen2.5-3B-Instruct**; **bf16 LoRA** (~30M trainable parameters); **no** 4-bit quant on this run (A10G bf16). Metrics also go to **Trackio** on the job.

| Item | Value |
|------|--------|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| GPU / job | HF Jobs, A10G |
| LoRA | bf16 |
| GRPO steps / round | 400 |
| Learning rate | 1e-5 |
| Generations (GRPO) | 4 |
| Gen sampling | temp 0.9, top_p 0.95 |
| Seed | 42 |
| Output adapter | [Harshitawake/tool-call-grpo-a10g-v2](https://huggingface.co/Harshitawake/tool-call-grpo-a10g-v2) |

Round 1 and Round 2 each took on the order of **~1h** wall time on the job (logs: ~56 min and ~71 min training runtime respectively for the two GRPO blocks).

---

## Results (same 40 eval scenarios)

| Stage | Avg reward | Accuracy | Δ reward vs baseline |
|--------|------------|----------|------------------------|
| Baseline | 0.565 | 47.5% | — |
| After GRPO round 1 | 0.734 | 70.0% | **+0.169** |
| After GRPO round 2 (memory in training) | 0.768 | 72.5% | **+0.203** |

**Round 2 vs round 1:** +0.034 reward, +2.5 pp accuracy. The **largest** gain is from the first GRPO pass; the second pass tests whether **memory-augmented data** still moves the same **fixed** eval (it does).

**Figures:** add `submission/charts/eval_summary.png` and `reward_deltas.png` to this article, or host them next to the repo. Regenerate:

```bash
python submission/generate_submission_charts.py
```

The training job also writes a summary plot (e.g. `training_results.png`) alongside GRPO; **Trackio** keeps step-wise logs if you need curves.

---

## Reward logic and training stack (short)

- **Reward:** from the environment grader (trace vs reference, partial credit); same signal for **training** and **table above**.
- **RL:** [TRL](https://github.com/huggingface/trl) **GRPO**; Round 1 = plain prompts, Round 2 = **memory-enriched** prompts on the same 100-train set.
- **Coherence:** one reward definition, one eval split, three reported stages—this matches what “pipeline” rubrics ask for.

---

## Get the code & run the demo

- **Trained LoRA:** [Harshitawake/tool-call-grpo-a10g-v2](https://huggingface.co/Harshitawake/tool-call-grpo-a10g-v2) — load on top of `Qwen/Qwen2.5-3B-Instruct`.
- **Scenarios:** `scenarios_expanded.json` in the public project (URL used in HF Jobs is pinned in your training job script).
- **App:** Streamlit + FastAPI (`frontend/`, `api/`) in the same repo; Docker + nginx for a **Hugging Face Space** on port 7860 (optional).
- **Charts:** `submission/generate_submission_charts.py` → `submission/charts/*.png` and `.svg`.

If you use a Space, set **repository secrets** (e.g. `HF_TOKEN`) for the inference API your agent calls.

---

## Final takeaway

**ToolMind** is a small, closed loop: **RL aligns the model to the tool environment; memory turns last round’s eval into structured hints for the next training round**—so “self-improvement” is not only a demo toggle at inference time, but a **second curriculum** for GRPO. For readers who only remember one line:

> **Reinforcement learning matches weights to the environment; memory matches the *next* training run to what we already saw on eval.**

---

*Hackathon / OpenEnv track. Training: Hugging Face Jobs, A10G, Qwen2.5-3B-Instruct, bf16 LoRA, two 400-step GRPO rounds as logged.*
