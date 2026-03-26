"""Vector indexers for embedding-based candidate generation."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MemoryIndexer(ABC):
    """Abstract base class for memory indexers."""

    @abstractmethod
    def index(self, memory_id: str, embedding: list[float]) -> None:
        """Add or update a memory in the index.

        Args:
            memory_id: Unique identifier for the memory.
            embedding: Vector embedding of the memory content.
        """
        pass

    @abstractmethod
    def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Search for similar memories.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.

        Returns:
            List of (memory_id, similarity_score) tuples, sorted by score descending.
        """
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """Remove a memory from the index.

        Args:
            memory_id: ID of the memory to remove.

        Returns:
            True if removed, False if not found.
        """
        pass

    @abstractmethod
    def reindex(self, memories: dict[str, list[float]]) -> None:
        """Rebuild the entire index from scratch.

        Args:
            memories: Dict mapping memory_id to embedding.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of indexed memories."""
        pass


class FAISSIndexer(MemoryIndexer):
    """FAISS-based indexer using IndexFlatIP (inner product = cosine on normalized vectors)."""

    def __init__(self, dimension: int = 384):
        """Initialize FAISS indexer.

        Args:
            dimension: Embedding dimension.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        self._next_idx = 0
        self._embeddings: dict[str, np.ndarray] = {}  # For reconstruction

    def index(self, memory_id: str, embedding: list[float]) -> None:
        """Add or update a memory in the index."""
        embedding_np = np.array(embedding, dtype=np.float32)

        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_np = embedding_np / norm

        embedding_np = embedding_np.reshape(1, -1)

        if memory_id in self._id_to_idx:
            # Update existing - FAISS doesn't support in-place update
            # So we just update our tracking; the old vector remains but won't be mapped
            pass

        # Add to index
        self._index.add(embedding_np)
        idx = self._next_idx
        self._id_to_idx[memory_id] = idx
        self._idx_to_id[idx] = memory_id
        self._embeddings[memory_id] = embedding_np.flatten()
        self._next_idx += 1

        logger.debug(f"Indexed memory {memory_id} at index {idx}")

    def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Search for similar memories."""
        if len(self._id_to_idx) == 0:
            return []

        query_np = np.array(query_embedding, dtype=np.float32)

        # Normalize
        norm = np.linalg.norm(query_np)
        if norm > 0:
            query_np = query_np / norm

        query_np = query_np.reshape(1, -1)

        # Search
        k = min(top_k, len(self._id_to_idx))
        scores, indices = self._index.search(query_np, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            memory_id = self._idx_to_id.get(int(idx))
            if memory_id:
                results.append((memory_id, float(score)))

        return results

    def remove(self, memory_id: str) -> bool:
        """Remove a memory from the index (marks as removed, doesn't actually delete)."""
        if memory_id not in self._id_to_idx:
            return False

        idx = self._id_to_idx.pop(memory_id)
        self._idx_to_id.pop(idx, None)
        self._embeddings.pop(memory_id, None)

        logger.debug(f"Removed memory {memory_id} from index")
        return True

    def reindex(self, memories: dict[str, list[float]]) -> None:
        """Rebuild the entire index from scratch."""
        import faiss

        self._index = faiss.IndexFlatIP(self._dimension)
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._embeddings.clear()
        self._next_idx = 0

        for memory_id, embedding in memories.items():
            self.index(memory_id, embedding)

        logger.info(f"Reindexed {len(memories)} memories")

    def __len__(self) -> int:
        """Return the number of indexed memories."""
        return len(self._id_to_idx)


class ChromaDBIndexer(MemoryIndexer):
    """ChromaDB-based indexer."""

    def __init__(
        self,
        collection_name: str = "memories",
        persist_directory: Optional[str] = None,
    ):
        """Initialize ChromaDB indexer.

        Args:
            collection_name: Name of the collection.
            persist_directory: Directory for persistence (None for in-memory).
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb is required. Install with: pip install chromadb")

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client(Settings(anonymized_telemetry=False))

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection_name = collection_name

    def index(self, memory_id: str, embedding: list[float]) -> None:
        """Add or update a memory in the index."""
        # ChromaDB handles upsert automatically
        self._collection.upsert(
            ids=[memory_id],
            embeddings=[embedding],
        )
        logger.debug(f"Indexed memory {memory_id}")

    def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Search for similar memories."""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
        )

        output = []
        if results["ids"] and results["distances"]:
            for memory_id, distance in zip(results["ids"][0], results["distances"][0]):
                # ChromaDB returns distance, convert to similarity
                # For cosine distance: similarity = 1 - distance
                similarity = 1 - distance
                output.append((memory_id, similarity))

        return output

    def remove(self, memory_id: str) -> bool:
        """Remove a memory from the index."""
        try:
            self._collection.delete(ids=[memory_id])
            logger.debug(f"Removed memory {memory_id}")
            return True
        except Exception:
            return False

    def reindex(self, memories: dict[str, list[float]]) -> None:
        """Rebuild the entire index from scratch."""
        # Delete and recreate collection
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        if memories:
            self._collection.add(
                ids=list(memories.keys()),
                embeddings=list(memories.values()),
            )

        logger.info(f"Reindexed {len(memories)} memories")

    def __len__(self) -> int:
        """Return the number of indexed memories."""
        return self._collection.count()
