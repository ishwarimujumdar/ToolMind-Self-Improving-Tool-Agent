"""
Agent API — FastAPI endpoints for demo and dashboard interaction.

Endpoints:
  POST /run-episode     - Run one episode and return metrics
  POST /run-comparison  - Run baseline vs memory comparison
  GET  /metrics         - Get training/episode history
  GET  /memory/stats    - Memory store statistics
  GET  /memory/search   - Search memory for lessons
  POST /memory/clear    - Clear memory store
  GET  /health          - Health check
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent))

from memory.memory_store import MemoryStore

app = FastAPI(title="ToolMind Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METRICS_FILE = DATA_DIR / "training_log.json"

memory_store = MemoryStore(persist_dir=str(DATA_DIR / "chroma_data"))


class EpisodeRequest(BaseModel):
    task_type: str = "hard"
    use_memory: bool = True
    episode_num: int = 0


class ComparisonRequest(BaseModel):
    task_type: str = "hard"
    num_episodes: int = 3


class MemorySearchRequest(BaseModel):
    query: str
    n_results: int = 3


def _load_metrics() -> list[dict]:
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            return json.load(f)
    return []


def _save_metrics(metrics: list[dict]):
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "memory_entries": memory_store.count(),
        "metrics_entries": len(_load_metrics()),
    }


@app.post("/run-episode")
def run_episode(req: EpisodeRequest):
    """Run a single episode and return results."""
    try:
        from agent.combined_agent import CombinedAgent

        agent = CombinedAgent(
            use_memory=req.use_memory,
            memory_dir=str(DATA_DIR / "chroma_data"),
        )
        result = agent.run_episode(
            task_type=req.task_type,
            episode_num=req.episode_num,
            verbose=False,
        )

        metrics = _load_metrics()
        metrics.append(result)
        _save_metrics(metrics)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-comparison")
def run_comparison(req: ComparisonRequest):
    """Run baseline vs memory comparison."""
    try:
        from agent.combined_agent import CombinedAgent

        agent = CombinedAgent(
            use_memory=True,
            memory_dir=str(DATA_DIR / "chroma_data"),
        )
        results = agent.run_comparison(
            task_type=req.task_type,
            num_episodes=req.num_episodes,
            verbose=False,
        )

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    """Get all logged training/episode metrics."""
    return _load_metrics()


@app.get("/memory/stats")
def memory_stats():
    """Get memory store statistics."""
    return memory_store.get_stats()


@app.post("/memory/search")
def memory_search(req: MemorySearchRequest):
    """Search memory for relevant lessons."""
    lessons = memory_store.retrieve_lessons(req.query, n_results=req.n_results)
    formatted = memory_store.format_lessons_for_prompt(req.query, n_results=req.n_results)
    return {
        "lessons": lessons,
        "formatted_prompt": formatted,
    }


@app.get("/memory/all")
def memory_all():
    """Get all stored experiences."""
    return memory_store.get_all_experiences(limit=200)


@app.post("/memory/clear")
def memory_clear():
    """Clear all memory."""
    memory_store.clear()
    return {"status": "cleared", "count": 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
