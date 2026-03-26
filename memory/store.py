"""Memory store for CRUD operations and persistence."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from memory.schema import Memory, MemoryFilter
from memory.indexer import MemoryIndexer, FAISSIndexer

logger = logging.getLogger(__name__)


class MemoryStore:
    """In-memory store with optional persistence for memories."""

    def __init__(
        self,
        indexer: Optional[MemoryIndexer] = None,
        persistence_path: Optional[str] = None,
        embedding_fn: Optional[callable] = None,
        supersession_threshold: float = 0.85,
        same_source_supersession: bool = True,
    ):
        """Initialize the memory store.

        Args:
            indexer: Vector indexer for similarity search. Defaults to FAISSIndexer.
            persistence_path: Path to JSON file for persistence. None for in-memory only.
            embedding_fn: Function to compute embeddings. Required for add operations.
            supersession_threshold: Similarity threshold for detecting supersession.
            same_source_supersession: Only detect supersession within same source.
        """
        self._memories: dict[str, Memory] = {}
        self._indexer = indexer or FAISSIndexer(dimension=384)
        self._persistence_path = Path(persistence_path) if persistence_path else None
        self._embedding_fn = embedding_fn
        self._supersession_threshold = supersession_threshold
        self._same_source_supersession = same_source_supersession

        if self._persistence_path and self._persistence_path.exists():
            self._load()

    def add(
        self,
        content: str,
        source: str,
        metadata: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Memory:
        """Add a new memory to the store.

        Args:
            content: Text content of the memory.
            source: Source identifier.
            metadata: Additional metadata.
            tags: Tags for categorization.
            timestamp: Optional timestamp (defaults to now).

        Returns:
            The created Memory object.

        Raises:
            ValueError: If embedding_fn is not configured.
        """
        if self._embedding_fn is None:
            raise ValueError("embedding_fn must be configured to add memories")

        # Compute embedding
        embedding = self._embedding_fn(content)

        # Create memory
        memory = Memory(
            content=content,
            source=source,
            embedding=embedding,
            metadata=metadata or {},
            tags=tags or [],
            timestamp=timestamp or datetime.utcnow(),
        )

        # Check for supersession
        superseded_id = self._check_supersession(memory)
        if superseded_id:
            superseded = self._memories[superseded_id]
            memory.supersedes = superseded_id
            memory.version = superseded.version + 1
            superseded.is_active = False
            logger.info(f"Memory {memory.id} supersedes {superseded_id}")

        # Store and index
        self._memories[memory.id] = memory
        self._indexer.index(memory.id, embedding)

        logger.debug(f"Added memory {memory.id}: {content[:50]}...")
        return memory

    def get(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID.

        Args:
            memory_id: The memory ID.

        Returns:
            The Memory object or None if not found.
        """
        return self._memories.get(memory_id)

    def update(self, memory_id: str, new_content: str) -> Memory:
        """Update a memory's content, creating a new version.

        Args:
            memory_id: ID of the memory to update.
            new_content: New content for the memory.

        Returns:
            The new Memory object (old one is marked inactive).

        Raises:
            KeyError: If memory not found.
            ValueError: If embedding_fn is not configured.
        """
        if memory_id not in self._memories:
            raise KeyError(f"Memory {memory_id} not found")

        if self._embedding_fn is None:
            raise ValueError("embedding_fn must be configured to update memories")

        old_memory = self._memories[memory_id]

        # Create new memory that supersedes the old one
        new_embedding = self._embedding_fn(new_content)
        new_memory = Memory(
            content=new_content,
            source=old_memory.source,
            embedding=new_embedding,
            metadata=old_memory.metadata.copy(),
            tags=old_memory.tags.copy(),
            version=old_memory.version + 1,
            supersedes=memory_id,
        )

        # Mark old as inactive
        old_memory.is_active = False

        # Store and index new
        self._memories[new_memory.id] = new_memory
        self._indexer.index(new_memory.id, new_embedding)

        logger.info(f"Updated memory {memory_id} -> {new_memory.id}")
        return new_memory

    def delete(self, memory_id: str) -> bool:
        """Soft delete a memory (set is_active=False).

        Args:
            memory_id: ID of the memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        if memory_id not in self._memories:
            return False

        self._memories[memory_id].is_active = False
        logger.info(f"Soft deleted memory {memory_id}")
        return True

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[MemoryFilter] = None,
    ) -> list[Memory]:
        """Search for similar memories.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            filters: Optional filters to apply.

        Returns:
            List of matching Memory objects, sorted by similarity.
        """
        # Get candidates from indexer (get more to account for filtering)
        search_k = top_k * 3 if filters else top_k
        candidates = self._indexer.search(query_embedding, search_k)

        results = []
        for memory_id, score in candidates:
            memory = self._memories.get(memory_id)
            if memory is None:
                continue

            # Apply filters
            if filters and not self._apply_filters(memory, filters):
                continue

            # Always filter inactive unless explicitly requested
            if filters is None or filters.require_active:
                if not memory.is_active:
                    continue

            # Update access stats
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()

            results.append(memory)
            if len(results) >= top_k:
                break

        return results

    def get_history(self, memory_id: str) -> list[Memory]:
        """Get the version history of a memory (supersession chain).

        Args:
            memory_id: ID of any memory in the chain.

        Returns:
            List of all versions, oldest first.
        """
        history = []
        current = self._memories.get(memory_id)

        if current is None:
            return []

        # Walk back to find the oldest version
        visited = set()
        while current and current.id not in visited:
            visited.add(current.id)
            history.append(current)
            if current.supersedes:
                current = self._memories.get(current.supersedes)
            else:
                break

        # Reverse to get oldest first
        history.reverse()

        # Also find any memories that supersede this one
        latest = history[-1] if history else None
        while latest:
            newer = None
            for m in self._memories.values():
                if m.supersedes == latest.id and m.id not in visited:
                    newer = m
                    break
            if newer:
                visited.add(newer.id)
                history.append(newer)
                latest = newer
            else:
                break

        return history

    def get_all(self, include_inactive: bool = False) -> list[Memory]:
        """Get all memories.

        Args:
            include_inactive: Whether to include inactive memories.

        Returns:
            List of all memories.
        """
        if include_inactive:
            return list(self._memories.values())
        return [m for m in self._memories.values() if m.is_active]

    def save(self) -> None:
        """Save memories to persistence file."""
        if self._persistence_path is None:
            logger.warning("No persistence path configured")
            return

        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            memory_id: memory.model_dump(mode="json")
            for memory_id, memory in self._memories.items()
        }

        with open(self._persistence_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved {len(data)} memories to {self._persistence_path}")

    def _load(self) -> None:
        """Load memories from persistence file."""
        if not self._persistence_path or not self._persistence_path.exists():
            return

        with open(self._persistence_path, "r") as f:
            data = json.load(f)

        for memory_id, memory_data in data.items():
            memory = Memory(**memory_data)
            self._memories[memory_id] = memory
            if memory.embedding:
                self._indexer.index(memory_id, memory.embedding)

        logger.info(f"Loaded {len(data)} memories from {self._persistence_path}")

    def _check_supersession(self, new_memory: Memory) -> Optional[str]:
        """Check if this memory supersedes an existing one.

        Args:
            new_memory: The new memory being added.

        Returns:
            ID of the superseded memory, or None.
        """
        if not new_memory.embedding:
            return None

        # Search for similar active memories
        candidates = self._indexer.search(new_memory.embedding, top_k=5)

        for memory_id, score in candidates:
            if score < self._supersession_threshold:
                continue

            existing = self._memories.get(memory_id)
            if existing is None or not existing.is_active:
                continue

            # Check same source requirement
            if self._same_source_supersession and existing.source != new_memory.source:
                continue

            # This memory supersedes the existing one
            return memory_id

        return None

    def _apply_filters(self, memory: Memory, filters: MemoryFilter) -> bool:
        """Apply filters to a memory.

        Args:
            memory: The memory to check.
            filters: Filters to apply.

        Returns:
            True if memory passes all filters.
        """
        # Time range filter
        if filters.time_range:
            if filters.time_range.start and memory.timestamp < filters.time_range.start:
                return False
            if filters.time_range.end and memory.timestamp > filters.time_range.end:
                return False

        # Source filter
        if filters.sources and memory.source not in filters.sources:
            return False

        # Tags filter (any match)
        if filters.tags:
            if not any(tag in memory.tags for tag in filters.tags):
                return False

        return True

    def __len__(self) -> int:
        """Return number of active memories."""
        return sum(1 for m in self._memories.values() if m.is_active)
