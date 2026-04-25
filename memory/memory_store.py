"""
Memory Store — ChromaDB-backed trajectory memory for self-improving tool agents.

Stores past episode experiences (query, tool sequence, reward, lesson)
and retrieves similar past experiences to guide future decisions.
"""

import json
import os
from datetime import datetime
from typing import Any

import chromadb
from chromadb.config import Settings


class MemoryStore:
    """Persistent memory for agent trajectories using ChromaDB."""

    def __init__(self, persist_dir: str = "./data/chroma_data", collection_name: str = "tool_experiences"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def store_experience(
        self,
        query: str,
        scenario_id: int,
        tool_sequence: list[str],
        reward: float,
        lesson: str,
        should_refuse: bool = False,
        difficulty: str = "medium",
        episode: int = 0,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store one episode's experience in memory."""
        entry_id = f"ep{episode}_s{scenario_id}_{datetime.now().strftime('%H%M%S')}"

        outcome = "correct" if reward > 0.7 else "partial" if reward > 0.3 else "wrong"

        metadata = {
            "scenario_id": str(scenario_id),
            "tool_sequence": json.dumps(tool_sequence),
            "reward": float(reward),
            "outcome": outcome,
            "lesson": lesson,
            "should_refuse": str(should_refuse),
            "difficulty": difficulty,
            "episode": str(episode),
            "timestamp": datetime.now().isoformat(),
        }
        if extra_metadata:
            for k, v in extra_metadata.items():
                metadata[k] = str(v)

        self.collection.upsert(
            documents=[query],
            metadatas=[metadata],
            ids=[entry_id],
        )
        return entry_id

    def retrieve_lessons(
        self,
        query: str,
        n_results: int = 3,
        min_reward: float | None = None,
    ) -> list[dict]:
        """Retrieve similar past experiences for a given query."""
        if self.count() == 0:
            return []

        n = min(n_results, self.count())
        results = self.collection.query(
            query_texts=[query],
            n_results=n,
        )

        experiences = []
        if not results["metadatas"] or not results["metadatas"][0]:
            return []

        for i, meta in enumerate(results["metadatas"][0]):
            reward = float(meta.get("reward", 0))
            if min_reward is not None and reward < min_reward:
                continue

            experiences.append({
                "query": results["documents"][0][i] if results["documents"] else "",
                "tool_sequence": json.loads(meta.get("tool_sequence", "[]")),
                "reward": reward,
                "outcome": meta.get("outcome", "unknown"),
                "lesson": meta.get("lesson", ""),
                "should_refuse": meta.get("should_refuse", "False") == "True",
                "similarity": results["distances"][0][i] if results["distances"] else 0.0,
                "difficulty": meta.get("difficulty", "medium"),
            })

        return experiences

    def get_tool_preference_scores(
        self,
        query: str,
        tool_names: list[str],
        n_results: int = 5,
    ) -> dict[str, float]:
        """Convert retrieved memories into per-tool preference scores."""
        experiences = self.retrieve_lessons(query, n_results=n_results)
        if not experiences:
            return {t: 0.0 for t in tool_names}

        scores: dict[str, float] = {t: 0.0 for t in tool_names}
        total_weight = 0.0

        for exp in experiences:
            sim = 1.0 - exp["similarity"]  # cosine distance → similarity
            reward = exp["reward"]

            if reward > 0.5:
                weight = sim * reward
            else:
                weight = sim * (reward - 1.0)

            for tool in exp["tool_sequence"]:
                if tool in scores:
                    scores[tool] += weight

            total_weight += abs(weight)

        if total_weight > 0:
            scores = {t: s / total_weight for t, s in scores.items()}

        return scores

    def format_lessons_for_prompt(
        self,
        query: str,
        n_results: int = 3,
    ) -> str:
        """Format retrieved lessons as a string for prompt injection."""
        experiences = self.retrieve_lessons(query, n_results=n_results)
        if not experiences:
            return ""

        positive = [e for e in experiences if e["reward"] > 0.5]
        negative = [e for e in experiences if e["reward"] <= 0.5]

        lines = []
        for exp in positive:
            tools = " → ".join(exp["tool_sequence"]) if exp["tool_sequence"] else "REFUSE"
            lines.append(
                f"  [reward={exp['reward']:.2f}] {exp['lesson']} (tools: {tools})"
            )

        for exp in negative:
            tools = " → ".join(exp["tool_sequence"]) if exp["tool_sequence"] else "REFUSE"
            lines.append(
                f"  [AVOID, reward={exp['reward']:.2f}] {exp['lesson']} (tools: {tools})"
            )

        if not lines:
            return ""

        return "LESSONS FROM PAST EXPERIENCE:\n" + "\n".join(lines)

    def get_all_experiences(self, limit: int = 100) -> list[dict]:
        """Get all stored experiences for analysis/export."""
        if self.count() == 0:
            return []

        results = self.collection.get(limit=limit, include=["documents", "metadatas"])
        experiences = []
        for i, meta in enumerate(results["metadatas"]):
            experiences.append({
                "id": results["ids"][i],
                "query": results["documents"][i],
                "tool_sequence": json.loads(meta.get("tool_sequence", "[]")),
                "reward": float(meta.get("reward", 0)),
                "outcome": meta.get("outcome", ""),
                "lesson": meta.get("lesson", ""),
                "episode": meta.get("episode", "0"),
                "difficulty": meta.get("difficulty", ""),
                "timestamp": meta.get("timestamp", ""),
            })
        return experiences

    def get_stats(self) -> dict:
        """Get summary statistics of the memory store."""
        if self.count() == 0:
            return {"total": 0, "avg_reward": 0.0, "correct": 0, "wrong": 0}

        all_exp = self.get_all_experiences(limit=1000)
        rewards = [e["reward"] for e in all_exp]
        return {
            "total": len(all_exp),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "correct": sum(1 for e in all_exp if e["outcome"] == "correct"),
            "partial": sum(1 for e in all_exp if e["outcome"] == "partial"),
            "wrong": sum(1 for e in all_exp if e["outcome"] == "wrong"),
            "episodes": len(set(e["episode"] for e in all_exp)),
        }

    def clear(self):
        """Clear all stored experiences."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
