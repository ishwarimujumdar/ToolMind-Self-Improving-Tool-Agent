# HuggingFace Jobs — Training Guide

Train your tool-call agent **2-4× faster** than Colab using HF Jobs with your $30 credits.

---

## Why HF Jobs?

| Platform           | GPU         | Speed     | Cost (per training run) |
| ------------------ | ----------- | --------- | ----------------------- |
| Colab Free         | T4 (16 GB)  | Baseline  | Free, but slow & flaky  |
| **HF Jobs (this)** | **A10G**    | **3-4×**  | **~$1-2 per full run**  |
| Colab Pro          | A100 (40GB) | 5×        | $10/mo subscription     |

A10G is **Ampere**, so it supports `bf16` (Colab T4 only does `fp16`). bf16 is more numerically stable for GRPO.

Your $30 of credits = **~30 hours of A10G-small** = many full training runs.

---

## Step 1 — Install the HF CLI (one time)

### Windows (PowerShell)
```powershell
pip install --upgrade huggingface_hub[cli]
```

### Mac/Linux
```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Verify:
```bash
hf --version
```

---

## Step 2 — Authenticate (one time)

You need a **token with WRITE permission** to push the trained adapter.

1. Go to <https://huggingface.co/settings/tokens>
2. Click **"Create new token"** → **Type: Write** → name it `jobs-train`
3. Copy the token, then run:

```bash
hf auth login
```
Paste the token when prompted.

---

## Step 3 — Verify billing is set up

HF Jobs requires a payment method on file (your $30 credits will be used first).

Check at: <https://huggingface.co/settings/billing>

---

## Step 4 — Submit your training job

Pick a mode based on how much demo polish you need:

### Quick smoke test (~25 min, ~$0.40)
```bash
hf jobs uv run \
    --flavor a10g-small \
    --secrets HF_TOKEN \
    --timeout 1h \
    https://raw.githubusercontent.com/Harshitawake/tool-call-rl-OpenEnv/main/training/grpo_hf_jobs.py \
    --mode fast \
    --rounds 1 \
    --output-repo YOUR_HF_USERNAME/tool-call-grpo-fast
```

### Demo-quality run (~60 min, ~$1.00) — **RECOMMENDED FOR HACKATHON**
```bash
hf jobs uv run \
    --flavor a10g-small \
    --secrets HF_TOKEN \
    --timeout 2h \
    https://raw.githubusercontent.com/Harshitawake/tool-call-rl-OpenEnv/main/training/grpo_hf_jobs.py \
    --mode demo \
    --rounds 2 \
    --output-repo YOUR_HF_USERNAME/tool-call-grpo-demo
```

### Full run (~90 min, ~$1.50)
```bash
hf jobs uv run \
    --flavor a10g-small \
    --secrets HF_TOKEN \
    --timeout 3h \
    https://raw.githubusercontent.com/Harshitawake/tool-call-rl-OpenEnv/main/training/grpo_hf_jobs.py \
    --mode full \
    --rounds 2 \
    --output-repo YOUR_HF_USERNAME/tool-call-grpo
```

> **Replace `YOUR_HF_USERNAME`** with your actual HuggingFace username (e.g. `harshitawake`).

The CLI prints a job ID and a URL like:
```
Job submitted: jobs/abc123
Logs: https://huggingface.co/jobs/abc123
```

---

## Step 5 — Monitor the run

```bash
# Live-stream logs in your terminal
hf jobs logs <job_id>

# Or list all your running jobs
hf jobs ps

# Inspect a specific job
hf jobs inspect <job_id>

# Cancel if you need to
hf jobs cancel <job_id>
```

You can also watch on the web: <https://huggingface.co/settings/jobs>

If `trackio` initialised successfully, you'll see live reward/loss curves on the **Trackio** dashboard at <https://huggingface.co/trackio>.

---

## Step 6 — When the job finishes

Your trained adapter is automatically pushed to:
```
https://huggingface.co/YOUR_HF_USERNAME/tool-call-grpo-demo
```

Along with:
- `adapter_config.json` and `adapter_model.safetensors` (the LoRA weights)
- `training_results.png` (3-panel plot: per-scenario, avg reward, accuracy)
- `results.json` (raw numbers for your demo)

---

## Step 7 — Use the trained model in your demo

Update your inference code to load the LoRA adapter from the Hub:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("YOUR_HF_USERNAME/tool-call-grpo-demo")
model = PeftModel.from_pretrained(base, "YOUR_HF_USERNAME/tool-call-grpo-demo")
```

Or merge the adapter and push as a full model for HF Spaces inference.

---

## Hyperparameters per mode

| Mode   | Train sc. | Eval sc. | Epochs | Generations | LR     | Time      |
| ------ | --------- | -------- | ------ | ----------- | ------ | --------- |
| `fast` | 30        | 20       | 1      | 4           | 3e-6   | ~25 min   |
| `demo` | 60        | 40       | 2      | 6           | 3e-6   | ~60 min   |
| `full` | 80        | 40       | 2      | 8           | 3e-6   | ~90 min   |

**Why these are better than the Colab defaults:**
- `bf16` instead of `fp16` (A10G Ampere supports it — more stable)
- More `num_generations` → smoother GRPO loss curve (bigger group = lower variance)
- Lower learning rate (3e-6 vs 5e-6) → less catastrophic forgetting
- 2 epochs instead of 1 → adapter actually converges

---

## Troubleshooting

**Job stuck in `PENDING`** → A10G capacity is finite. Try `t4-medium` (cheaper, still 2× faster than free Colab):
```bash
--flavor t4-medium
```

**Push to Hub fails at the end** → Check that your `HF_TOKEN` has **Write** permission and the `--output-repo` namespace matches your account.

**Out of memory** → Reduce `--mode` (try `fast`) or use `a10g-large` (more VRAM, more $).

**Job times out** → Increase `--timeout` (e.g. `--timeout 4h`).

**Want to retrain with different params** → Just re-run with a different `--output-repo` name. Each run is fully isolated.

---

## Cost estimates (with current HF pricing)

| Flavor       | $/hr    | Demo run cost | Full run cost |
| ------------ | ------- | ------------- | ------------- |
| `t4-small`   | ~$0.40  | ~$0.40        | ~$0.60        |
| `t4-medium`  | ~$0.60  | ~$0.60        | ~$0.90        |
| `a10g-small` | ~$1.00  | **~$1.00**    | ~$1.50        |
| `a10g-large` | ~$3.00  | ~$3.00        | ~$4.50        |

Your **$30 credits → ~30 hours of A10G-small → ~30 demo runs**. Plenty of room to iterate.
