"""High-level retrieval pipeline API."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from agents.orchestrator import Orchestrator
from memory.schema import Memory, MemoryQuery, RetrievalResult
from memory.store import MemoryStore
from retrieval.candidate import CandidateGenerator
from retrieval.context_builder import ContextBuilder

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """High-level API for ASMR memory retrieval."""

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        orchestrator: Optional[Orchestrator] = None,
        candidate_generator: Optional[CandidateGenerator] = None,
        context_builder: Optional[ContextBuilder] = None,
        embedding_fn: Optional[callable] = None,
        candidate_top_k: int = 50,
    ):
        """Initialize the retrieval pipeline.

        Args:
            store: MemoryStore instance (or creates default).
            orchestrator: Orchestrator instance (or creates default).
            candidate_generator: CandidateGenerator instance (or creates default).
            context_builder: ContextBuilder instance (or creates default).
            embedding_fn: Function to compute embeddings. Uses default if not provided.
            candidate_top_k: Number of candidates to retrieve for agent processing.
        """
        self._embedding_fn = embedding_fn or self._default_embedding_fn()

        self._store = store or MemoryStore(embedding_fn=self._embedding_fn)
        self._orchestrator = orchestrator or Orchestrator()
        self._candidate_generator = candidate_generator or CandidateGenerator(
            store=self._store,
            embedding_fn=self._embedding_fn,
        )
        self._context_builder = context_builder or ContextBuilder()
        self._candidate_top_k = candidate_top_k

    def _default_embedding_fn(self) -> callable:
        """Create default embedding function using sentence-transformers.

        Returns:
            Embedding function.
        """
        model = None

        def embed(text: str) -> list[float]:
            nonlocal model
            if model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer("all-MiniLM-L6-v2")
                except ImportError:
                    raise ImportError(
                        "sentence-transformers required. Install with: pip install sentence-transformers"
                    )
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

        return embed

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        require_reasoning: bool = True,
        filters: Optional[dict] = None,
    ) -> RetrievalResult:
        """Retrieve memories relevant to a query.

        This is the main entry point for ASMR retrieval.

        Args:
            query: The search query.
            top_k: Number of memories to return.
            require_reasoning: Whether to use agent reasoning.
            filters: Optional filters (time_range, sources, tags).

        Returns:
            RetrievalResult with memories, decisions, and context.
        """
        return asyncio.run(
            self.aretrieve(query, top_k, require_reasoning, filters)
        )

    async def aretrieve(
        self,
        query: str,
        top_k: int = 5,
        require_reasoning: bool = True,
        filters: Optional[dict] = None,
    ) -> RetrievalResult:
        """Async version of retrieve.

        Args:
            query: The search query.
            top_k: Number of memories to return.
            require_reasoning: Whether to use agent reasoning.
            filters: Optional filters.

        Returns:
            RetrievalResult.
        """
        # Create query object
        memory_query = MemoryQuery(
            query=query,
            top_k=top_k,
            require_reasoning=require_reasoning,
            filters=filters,
        )

        # Generate candidates
        candidates = self._candidate_generator.generate(
            query=query,
            top_k=self._candidate_top_k,
            filters=filters,
        )

        if not candidates:
            logger.info("No candidates found for query")
            return RetrievalResult(
                memories=[],
                agent_decisions=[],
                final_context="",
                metadata={"candidates_considered": 0},
            )

        # Run agent pipeline
        result = await self._orchestrator.run(memory_query, candidates)

        # Limit to requested top_k
        if len(result.memories) > top_k:
            result.memories = result.memories[:top_k]

        return result

    def retrieve_fast(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> RetrievalResult:
        """Fast retrieval without agent reasoning.

        Uses embedding similarity only, no LLM calls.

        Args:
            query: The search query.
            top_k: Number of memories to return.
            filters: Optional filters.

        Returns:
            RetrievalResult with memories but no agent decisions.
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            require_reasoning=False,
            filters=filters,
        )

    def add_memory(
        self,
        content: str,
        source: str,
        metadata: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Memory:
        """Add a memory to the store.

        Args:
            content: Text content of the memory.
            source: Source identifier.
            metadata: Additional metadata.
            tags: Tags for categorization.
            timestamp: Optional timestamp (defaults to now).

        Returns:
            The created Memory object.
        """
        return self._store.add(
            content=content,
            source=source,
            metadata=metadata,
            tags=tags,
            timestamp=timestamp,
        )

    def update_memory(self, memory_id: str, new_content: str) -> Memory:
        """Update a memory's content.

        Creates a new version that supersedes the old one.

        Args:
            memory_id: ID of memory to update.
            new_content: New content.

        Returns:
            The new Memory object.
        """
        return self._store.update(memory_id, new_content)

    def delete_memory(self, memory_id: str) -> bool:
        """Soft delete a memory.

        Args:
            memory_id: ID of memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        return self._store.delete(memory_id)

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID.

        Args:
            memory_id: Memory ID.

        Returns:
            Memory object or None.
        """
        return self._store.get(memory_id)

    def get_memory_history(self, memory_id: str) -> list[Memory]:
        """Get version history of a memory.

        Args:
            memory_id: ID of any memory in the chain.

        Returns:
            List of all versions, oldest first.
        """
        return self._store.get_history(memory_id)

    def list_memories(self, include_inactive: bool = False) -> list[Memory]:
        """List all memories.

        Args:
            include_inactive: Whether to include inactive memories.

        Returns:
            List of memories.
        """
        return self._store.get_all(include_inactive=include_inactive)

    def save(self) -> None:
        """Save memories to persistence file."""
        self._store.save()

    @property
    def store(self) -> MemoryStore:
        """Get the underlying memory store."""
        return self._store

    @property
    def orchestrator(self) -> Orchestrator:
        """Get the orchestrator."""
        return self._orchestrator

    def __len__(self) -> int:
        """Return number of active memories."""
        return len(self._store)
