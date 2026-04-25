"""
ToolMind Dashboard — Streamlit app for visualizing agent improvement.

Shows:
  - Learning curves (reward over episodes)
  - Memory statistics and browser
  - Live episode runner
  - Before/after comparison
"""

import json
import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METRICS_FILE = DATA_DIR / "training_log.json"


def load_metrics():
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            return json.load(f)
    return []


def load_memory_stats():
    try:
        from memory.memory_store import MemoryStore
        store = MemoryStore(persist_dir=str(DATA_DIR / "chroma_data"))
        return store.get_stats(), store
    except Exception:
        return {"total": 0, "avg_reward": 0, "correct": 0, "wrong": 0, "partial": 0, "episodes": 0}, None


st.set_page_config(
    page_title="ToolMind: Self-Improving Agent",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 ToolMind: Self-Improving Tool Agent")
st.markdown(
    "> *GRPO trains the weights. Memory trains the behavior. Together, the agent never stops improving.*"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Learning Curves",
    "🧠 Memory Explorer",
    "🚀 Run Demo",
    "📋 Architecture",
])

# ================================================================
# TAB 1: Learning Curves
# ================================================================
with tab1:
    metrics = load_metrics()

    if not metrics:
        st.info("No training data yet. Run some episodes to see learning curves.")

        st.subheader("Expected Results After Training")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Baseline (untrained)", "GRPO Round 1", "GRPO Round 2", "GRPO + Memory"],
            y=[0.45, 0.72, 0.82, 0.90],
            marker_color=["#ff6b6b", "#feca57", "#48dbfb", "#0abde3"],
            text=["0.45", "0.72", "0.82", "0.90"],
            textposition="auto",
        ))
        fig.update_layout(
            title="Expected Training Progress",
            yaxis_title="Average Reward",
            yaxis_range=[0, 1],
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        col1, col2, col3, col4 = st.columns(4)

        avg_rewards = [m.get("avg_reward", 0) for m in metrics]
        accuracies = [m.get("accuracy", 0) for m in metrics]

        col1.metric("Episodes Run", len(metrics))
        col2.metric("Latest Reward", f"{avg_rewards[-1]:.3f}" if avg_rewards else "N/A")
        col3.metric("Best Reward", f"{max(avg_rewards):.3f}" if avg_rewards else "N/A")
        col4.metric("Latest Accuracy", f"{accuracies[-1]:.1%}" if accuracies else "N/A")

        fig_reward = go.Figure()
        fig_reward.add_trace(go.Scatter(
            x=list(range(1, len(avg_rewards) + 1)),
            y=avg_rewards,
            mode="lines+markers",
            name="Avg Reward",
            line=dict(color="#0abde3", width=3),
        ))
        fig_reward.update_layout(
            title="Average Reward per Episode",
            xaxis_title="Episode",
            yaxis_title="Average Reward",
            yaxis_range=[0, 1],
            height=400,
        )
        st.plotly_chart(fig_reward, use_container_width=True)

        baseline = [m for m in metrics if not m.get("use_memory", True)]
        with_mem = [m for m in metrics if m.get("use_memory", True)]

        if baseline and with_mem:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name="Without Memory",
                x=[f"Ep {m.get('episode', i)}" for i, m in enumerate(baseline)],
                y=[m.get("avg_reward", 0) for m in baseline],
                marker_color="#ff6b6b",
            ))
            fig_comp.add_trace(go.Bar(
                name="With Memory",
                x=[f"Ep {m.get('episode', i)}" for i, m in enumerate(with_mem)],
                y=[m.get("avg_reward", 0) for m in with_mem],
                marker_color="#0abde3",
            ))
            fig_comp.update_layout(
                title="Baseline vs Memory-Augmented",
                yaxis_title="Average Reward",
                barmode="group",
                height=400,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        if len(metrics) > 1:
            all_rewards = []
            for m in metrics:
                if "rewards" in m:
                    for i, r in enumerate(m["rewards"]):
                        all_rewards.append({
                            "Episode": m.get("episode", 0),
                            "Scenario": i + 1,
                            "Reward": r,
                        })

            if all_rewards:
                import pandas as pd
                df = pd.DataFrame(all_rewards)
                fig_heat = px.density_heatmap(
                    df, x="Scenario", y="Episode", z="Reward",
                    title="Reward Heatmap (Episode x Scenario)",
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig_heat, use_container_width=True)

# ================================================================
# TAB 2: Memory Explorer
# ================================================================
with tab2:
    mem_stats, mem_store = load_memory_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Memories", mem_stats.get("total", 0))
    col2.metric("Avg Reward", f"{mem_stats.get('avg_reward', 0):.3f}")
    col3.metric("Correct", mem_stats.get("correct", 0))
    col4.metric("Wrong", mem_stats.get("wrong", 0))

    st.subheader("Search Memory")
    search_query = st.text_input("Enter a query to find relevant lessons:", "What's the weather in Tokyo?")

    if st.button("Search") and mem_store:
        lessons = mem_store.retrieve_lessons(search_query, n_results=5)
        if lessons:
            for i, lesson in enumerate(lessons):
                tools = " → ".join(lesson["tool_sequence"]) if lesson["tool_sequence"] else "REFUSED"
                reward_color = "green" if lesson["reward"] > 0.7 else "orange" if lesson["reward"] > 0.3 else "red"
                st.markdown(
                    f"**{i+1}.** [{lesson['outcome']}] "
                    f"Reward: :{reward_color}[{lesson['reward']:.2f}] | "
                    f"Tools: `{tools}` | "
                    f"Lesson: *{lesson['lesson']}*"
                )
        else:
            st.info("No memories found. Run some episodes first.")

        formatted = mem_store.format_lessons_for_prompt(search_query, n_results=3)
        if formatted:
            st.subheader("Prompt Injection Preview")
            st.code(formatted, language="text")

    st.subheader("All Memories")
    if mem_store and mem_store.count() > 0:
        all_exp = mem_store.get_all_experiences(limit=50)
        for exp in all_exp:
            tools = " → ".join(exp["tool_sequence"]) if exp["tool_sequence"] else "REFUSED"
            st.text(
                f"[{exp['outcome']:7s}] reward={exp['reward']:.2f} | {tools:40s} | {exp['query'][:60]}"
            )
    else:
        st.info("Memory is empty. Run episodes to populate.")

    if st.button("Clear Memory") and mem_store:
        mem_store.clear()
        st.success("Memory cleared.")
        st.rerun()

# ================================================================
# TAB 3: Run Demo
# ================================================================
with tab3:
    st.subheader("Run an Episode")
    st.markdown("Run episodes to see the agent improve over time. Memory accumulates across runs.")

    col1, col2, col3 = st.columns(3)
    task_type = col1.selectbox("Difficulty", ["easy", "medium", "hard"], index=2)
    use_memory = col2.checkbox("Use Memory", value=True)
    episode_num = col3.number_input("Episode #", min_value=0, value=len(load_metrics()), step=1)

    if st.button("Run Episode", type="primary"):
        with st.spinner("Running episode... (this calls the LLM API)"):
            try:
                from agent.combined_agent import CombinedAgent

                agent = CombinedAgent(
                    use_memory=use_memory,
                    memory_dir=str(DATA_DIR / "chroma_data"),
                )
                result = agent.run_episode(
                    task_type=task_type,
                    episode_num=episode_num,
                    verbose=False,
                )

                metrics = load_metrics()
                metrics.append(result)
                with open(METRICS_FILE, "w") as f:
                    json.dump(metrics, f, indent=2)

                st.success(f"Episode complete! Avg reward: {result['avg_reward']:.3f} | Accuracy: {result['accuracy']:.1%}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Reward", f"{result['avg_reward']:.3f}")
                col2.metric("Accuracy", f"{result['accuracy']:.1%}")
                col3.metric("Memory Size", result.get("memory_size", 0))

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(range(1, len(result["rewards"]) + 1)),
                    y=result["rewards"],
                    marker_color=["#0abde3" if r > 0.7 else "#feca57" if r > 0.3 else "#ff6b6b" for r in result["rewards"]],
                ))
                fig.update_layout(
                    title="Per-Scenario Rewards",
                    xaxis_title="Scenario",
                    yaxis_title="Reward",
                    yaxis_range=[0, 1],
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ================================================================
# TAB 4: Architecture
# ================================================================
with tab4:
    st.subheader("System Architecture")
    st.markdown("""
### How ToolMind Works

**Training Phase (GRPO via TRL + Unsloth)**
```
Scenarios → GRPO Training → Better LLM Weights
                ↓
        Collect Experiences
                ↓
        Store in Memory (ChromaDB)
                ↓
        GRPO Round 2 (lesson-enriched prompts)
                ↓
        Even Better LLM Weights
```

**Inference Phase (Self-Improving)**
```
New Query
    ↓
Memory Retrieval (ChromaDB)
    → "Past lessons for similar queries"
    ↓
GRPO-Trained LLM
    → Generates tool_calls JSON
    ↓
Environment Grades Action
    → Reward (0.0 - 1.0)
    ↓
Store to Memory
    → Lesson for future queries
```

### Key Innovation

> GRPO trains the weights. Memory trains the behavior.
> Together, the agent never stops improving.

**Stage 1:** Baseline model — no training, no memory
**Stage 2:** GRPO Round 1 — trained without lessons
**Stage 3:** GRPO Round 2 — trained WITH lessons from memory
**Stage 4:** Inference with memory — keeps improving without retraining

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Environment | OpenEnv (ToolCallEnv) | Verifiable reward via grading |
| Training | TRL GRPOTrainer + Unsloth | RL weight updates |
| Memory | ChromaDB | Trajectory storage & retrieval |
| Model | Qwen2.5-3B (train) / 7B (deploy) | Policy (tool selection) |
| Dashboard | Streamlit | Visualization & demo |
    """)

    st.subheader("Environment Details")
    st.markdown("""
- **25 base scenarios** (expandable to 150+ via generator)
- **16 tools**: get_weather, search_flights, send_email, calculator, etc.
- **3 difficulty tiers**: Easy (tool selection), Medium (+params), Hard (+ordering+safety)
- **Verifiable rewards**: Rule-based grading, no learned reward model (RLVR)
    """)
