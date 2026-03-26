"""Baseline retrieval methods for comparison."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from memory.schema import Memory


class BaselineRetriever(ABC):
    """Abstract base class for baseline retrievers."""

    @abstractmethod
    def retrieve(
        self,
        query_embedding: list[float],
        memories: list[Memory],
        top_k: int = 5,
    ) -> list[Memory]:
        """Retrieve top-k memories.

        Args:
            query_embedding: Query vector.
            memories: All memories to search.
            top_k: Number of results.

        Returns:
            List of top-k memories.
        """
        pass


class NaiveRAGRetriever(BaselineRetriever):
    """Pure cosine similarity retrieval (standard RAG)."""

    def retrieve(
        self,
        query_embedding: list[float],
        memories: list[Memory],
        top_k: int = 5,
    ) -> list[Memory]:
        """Retrieve by pure cosine similarity.

        Args:
            query_embedding: Query vector.
            memories: All memories to search.
            top_k: Number of results.

        Returns:
            Top-k memories by cosine similarity.
        """
        if not memories:
            return []

        query = np.array(query_embedding)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        scored = []
        for mem in memories:
            if mem.embedding is None or not mem.is_active:
                continue

            mem_vec = np.array(mem.embedding)
            mem_norm = np.linalg.norm(mem_vec)
            if mem_norm > 0:
                mem_vec = mem_vec / mem_norm

            score = np.dot(query, mem_vec)
            scored.append((mem, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return [m for m, _ in scored[:top_k]]


class MMRRetriever(BaselineRetriever):
    """Maximal Marginal Relevance retrieval (diversity-aware)."""

    def __init__(self, lambda_param: float = 0.5):
        """Initialize MMR retriever.

        Args:
            lambda_param: Balance between relevance and diversity (0-1).
                         Higher = more relevance, lower = more diversity.
        """
        self.lambda_param = lambda_param

    def retrieve(
        self,
        query_embedding: list[float],
        memories: list[Memory],
        top_k: int = 5,
    ) -> list[Memory]:
        """Retrieve using Maximal Marginal Relevance.

        Args:
            query_embedding: Query vector.
            memories: All memories to search.
            top_k: Number of results.

        Returns:
            Top-k memories with diversity.
        """
        if not memories:
            return []

        # Filter memories with embeddings
        valid_memories = [m for m in memories if m.embedding and m.is_active]
        if not valid_memories:
            return []

        query = np.array(query_embedding)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        # Precompute similarities to query
        query_sims = {}
        embeddings = {}
        for mem in valid_memories:
            vec = np.array(mem.embedding)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings[mem.id] = vec
            query_sims[mem.id] = np.dot(query, vec)

        selected = []
        remaining = set(m.id for m in valid_memories)
        id_to_mem = {m.id: m for m in valid_memories}

        while len(selected) < top_k and remaining:
            best_id = None
            best_score = -float("inf")

            for mem_id in remaining:
                # Relevance to query
                relevance = query_sims[mem_id]

                # Maximum similarity to already selected
                max_sim = 0.0
                for sel_id in selected:
                    sim = np.dot(embeddings[mem_id], embeddings[sel_id])
                    max_sim = max(max_sim, sim)

                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim

                if mmr > best_score:
                    best_score = mmr
                    best_id = mem_id

            if best_id:
                selected.append(best_id)
                remaining.remove(best_id)

        return [id_to_mem[id] for id in selected]


class TimeWeightedRetriever(BaselineRetriever):
    """Cosine similarity weighted by recency (exponential decay)."""

    def __init__(self, half_life_days: float = 30.0, recency_weight: float = 0.3):
        """Initialize time-weighted retriever.

        Args:
            half_life_days: Half-life for recency decay.
            recency_weight: Weight of recency in final score (0-1).
        """
        self.half_life_days = half_life_days
        self.recency_weight = recency_weight

    def retrieve(
        self,
        query_embedding: list[float],
        memories: list[Memory],
        top_k: int = 5,
        reference_time: Optional["datetime"] = None,
    ) -> list[Memory]:
        """Retrieve by similarity * recency.

        Args:
            query_embedding: Query vector.
            memories: All memories to search.
            top_k: Number of results.
            reference_time: Reference time for recency (default: now).

        Returns:
            Top-k memories by combined score.
        """
        from datetime import datetime

        if not memories:
            return []

        ref_time = reference_time or datetime.utcnow()

        query = np.array(query_embedding)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        scored = []
        for mem in memories:
            if mem.embedding is None or not mem.is_active:
                continue

            # Cosine similarity
            mem_vec = np.array(mem.embedding)
            mem_norm = np.linalg.norm(mem_vec)
            if mem_norm > 0:
                mem_vec = mem_vec / mem_norm
            similarity = np.dot(query, mem_vec)

            # Recency score (exponential decay)
            age_days = (ref_time - mem.timestamp).total_seconds() / 86400
            recency = 0.5 ** (age_days / self.half_life_days)

            # Combined score
            score = (
                (1 - self.recency_weight) * similarity
                + self.recency_weight * recency
            )
            scored.append((mem, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [m for m, _ in scored[:top_k]]


class HybridRetriever(BaselineRetriever):
    """Hybrid retrieval combining multiple signals."""

    def __init__(
        self,
        similarity_weight: float = 0.5,
        recency_weight: float = 0.3,
        popularity_weight: float = 0.2,
        half_life_days: float = 30.0,
    ):
        """Initialize hybrid retriever.

        Args:
            similarity_weight: Weight for cosine similarity.
            recency_weight: Weight for recency.
            popularity_weight: Weight for access count.
            half_life_days: Half-life for recency decay.
        """
        self.similarity_weight = similarity_weight
        self.recency_weight = recency_weight
        self.popularity_weight = popularity_weight
        self.half_life_days = half_life_days

    def retrieve(
        self,
        query_embedding: list[float],
        memories: list[Memory],
        top_k: int = 5,
    ) -> list[Memory]:
        """Retrieve using hybrid scoring.

        Args:
            query_embedding: Query vector.
            memories: All memories to search.
            top_k: Number of results.

        Returns:
            Top-k memories by hybrid score.
        """
        from datetime import datetime

        if not memories:
            return []

        ref_time = datetime.utcnow()

        query = np.array(query_embedding)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        # Compute max access count for normalization
        max_access = max((m.access_count for m in memories), default=1) or 1

        scored = []
        for mem in memories:
            if mem.embedding is None or not mem.is_active:
                continue

            # Similarity
            mem_vec = np.array(mem.embedding)
            mem_norm = np.linalg.norm(mem_vec)
            if mem_norm > 0:
                mem_vec = mem_vec / mem_norm
            similarity = np.dot(query, mem_vec)

            # Recency
            age_days = (ref_time - mem.timestamp).total_seconds() / 86400
            recency = 0.5 ** (age_days / self.half_life_days)

            # Popularity (normalized)
            popularity = mem.access_count / max_access

            # Combined score
            score = (
                self.similarity_weight * similarity
                + self.recency_weight * recency
                + self.popularity_weight * popularity
            )
            scored.append((mem, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [m for m, _ in scored[:top_k]]
