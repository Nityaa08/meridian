import asyncio
import threading
import numpy as np
from app.agents.base import BaseAgent
from app.config import settings


# Serialise encode() calls across sessions too, so concurrent research
# requests never run the model in two threads at once.
_ENCODE_LOCK = threading.Lock()

# Default relevance threshold (cosine similarity) for sentence-transformers embeddings.
MIN_SIMILARITY = 0.15


def _encode(embedder, texts: list[str]):
    with _ENCODE_LOCK:
        return embedder.encode(texts)


class ExtractorAgent(BaseAgent):
    def __init__(self, embedder):
        super().__init__("ExtractorAgent")
        self.embedder = embedder

    async def run(self, query: str, passages: list[dict], queue: asyncio.Queue) -> list[dict]:
        await self.emit(queue, "EXTRACTING", f"Ranking {len(passages)} passages by semantic relevance")

        if not passages:
            await self.emit(queue, "EXTRACTING", "No passages to rank")
            return []

        texts = [p["text"] for p in passages]

        # One encode call for the query and all passages. Two concurrent
        # encode() calls in separate threads segfaulted on Apple Silicon (MPS).
        embeddings = await asyncio.to_thread(_encode, self.embedder, [query] + texts)
        query_vec = embeddings[0]
        passage_embeddings = embeddings[1:]
        similarities = []
        for i, pv in enumerate(passage_embeddings):
            sim = float(np.dot(query_vec, pv) / (np.linalg.norm(query_vec) * np.linalg.norm(pv) + 1e-9))
            similarities.append(sim)

        ranked = sorted(
            [{"similarity": sim, **passages[i]} for i, sim in enumerate(similarities)],
            key=lambda x: x["similarity"],
            reverse=True,
        )

        # Drop passages below minimum similarity threshold — removes junk results.
        # The threshold depends on the embedder: lexical hashing vectors give much
        # lower cosines than MiniLM for the same match, so they set their own.
        min_similarity = getattr(self.embedder, "min_similarity", MIN_SIMILARITY)
        ranked = [p for p in ranked if p["similarity"] >= min_similarity]

        top_k = ranked[:settings.top_k_passages]

        if not top_k:
            await self.emit(queue, "EXTRACTING", "No passages met the relevance threshold")
            return []

        avg_sim = float(np.mean([p["similarity"] for p in top_k]))
        await self.emit(
            queue, "EXTRACTING",
            f"Selected top {len(top_k)} passages (avg similarity: {avg_sim:.3f})",
            {"top_k": len(top_k), "avg_similarity": avg_sim},
        )
        return top_k
