"""Candidate generation for memory retrieval."""

import logging
from typing import Optional

from memory.schema import Memory, MemoryFilter, TimeRange
from memory.store import MemoryStore

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """Generates candidate memories from embedding similarity search."""

    def __init__(
        self,
        store: MemoryStore,
        embedding_fn: callable,
        default_top_k: int = 50,
    ):
        """Initialize the candidate generator.

        Args:
            store: MemoryStore to search.
            embedding_fn: Function to compute query embeddings.
            default_top_k: Default number of candidates to generate.
        """
        self._store = store
        self._embedding_fn = embedding_fn
        self._default_top_k = default_top_k

    def generate(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> list[Memory]:
        """Generate candidate memories for a query.

        Args:
            query: The search query.
            top_k: Number of candidates to return.
            filters: Optional filters to apply.

        Returns:
            List of candidate Memory objects.
        """
        k = top_k or self._default_top_k

        # Compute query embedding
        query_embedding = self._embedding_fn(query)

        # Convert dict filters to MemoryFilter if provided
        memory_filter = None
        if filters:
            memory_filter = self._parse_filters(filters)

        # Search store
        candidates = self._store.search(
            query_embedding=query_embedding,
            top_k=k,
            filters=memory_filter,
        )

        logger.info(f"Generated {len(candidates)} candidates for query: {query[:50]}...")
        return candidates

    def _parse_filters(self, filters: dict) -> MemoryFilter:
        """Parse filter dict into MemoryFilter object.

        Args:
            filters: Dict with filter parameters.

        Returns:
            MemoryFilter object.
        """
        time_range = None
        if "time_range" in filters:
            tr = filters["time_range"]
            time_range = TimeRange(
                start=tr.get("start"),
                end=tr.get("end"),
            )

        return MemoryFilter(
            time_range=time_range,
            sources=filters.get("sources"),
            tags=filters.get("tags"),
            require_active=filters.get("require_active", True),
        )

    def generate_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> list[tuple[Memory, float]]:
        """Generate candidates with similarity scores.

        Args:
            query: The search query.
            top_k: Number of candidates.
            filters: Optional filters.

        Returns:
            List of (Memory, similarity_score) tuples.
        """
        k = top_k or self._default_top_k
        query_embedding = self._embedding_fn(query)

        # Get raw search results with scores
        memory_filter = self._parse_filters(filters) if filters else None

        # Search in indexer directly for scores
        indexer = self._store._indexer
        raw_results = indexer.search(query_embedding, k * 2)  # Get more to filter

        results = []
        for memory_id, score in raw_results:
            memory = self._store.get(memory_id)
            if memory is None or not memory.is_active:
                continue

            if memory_filter and not self._apply_filter(memory, memory_filter):
                continue

            results.append((memory, score))
            if len(results) >= k:
                break

        return results

    def _apply_filter(self, memory: Memory, filters: MemoryFilter) -> bool:
        """Apply filters to a memory.

        Args:
            memory: Memory to check.
            filters: Filters to apply.

        Returns:
            True if memory passes filters.
        """
        if filters.time_range:
            if filters.time_range.start and memory.timestamp < filters.time_range.start:
                return False
            if filters.time_range.end and memory.timestamp > filters.time_range.end:
                return False

        if filters.sources and memory.source not in filters.sources:
            return False

        if filters.tags and not any(t in memory.tags for t in filters.tags):
            return False

        return True
