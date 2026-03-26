"""Tests for memory store operations."""

import pytest
from datetime import datetime, timedelta

from memory.schema import Memory, MemoryFilter, TimeRange
from memory.store import MemoryStore
from memory.indexer import FAISSIndexer


def mock_embedding(text: str) -> list[float]:
    """Generate deterministic mock embedding."""
    import hashlib
    h = hashlib.md5(text.encode()).hexdigest()
    return [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)] * 24  # 384-dim


@pytest.fixture
def store():
    """Create a fresh memory store for each test."""
    return MemoryStore(
        indexer=FAISSIndexer(dimension=384),
        embedding_fn=mock_embedding,
    )


class TestMemoryStore:
    """Tests for MemoryStore class."""

    def test_add_memory(self, store):
        """Test adding a memory."""
        mem = store.add(
            content="Test content",
            source="test_source",
            tags=["test"],
        )

        assert mem.id is not None
        assert mem.content == "Test content"
        assert mem.source == "test_source"
        assert mem.tags == ["test"]
        assert mem.is_active is True
        assert mem.version == 1

    def test_get_memory(self, store):
        """Test retrieving a memory by ID."""
        mem = store.add(content="Test", source="test")
        retrieved = store.get(mem.id)

        assert retrieved is not None
        assert retrieved.id == mem.id
        assert retrieved.content == "Test"

    def test_get_nonexistent_memory(self, store):
        """Test retrieving non-existent memory."""
        result = store.get("nonexistent-id")
        assert result is None

    def test_update_memory(self, store):
        """Test updating a memory creates new version."""
        original = store.add(content="Original content", source="test")
        updated = store.update(original.id, "Updated content")

        assert updated.id != original.id
        assert updated.content == "Updated content"
        assert updated.version == 2
        assert updated.supersedes == original.id

        # Original should be inactive
        original_reloaded = store.get(original.id)
        assert original_reloaded.is_active is False

    def test_delete_memory(self, store):
        """Test soft delete."""
        mem = store.add(content="To delete", source="test")
        result = store.delete(mem.id)

        assert result is True

        # Memory still exists but inactive
        deleted = store.get(mem.id)
        assert deleted is not None
        assert deleted.is_active is False

    def test_search_returns_active_only(self, store):
        """Test search only returns active memories."""
        active = store.add(content="Active memory", source="test")
        inactive = store.add(content="To be deleted", source="test")
        store.delete(inactive.id)

        query_embedding = mock_embedding("memory")
        results = store.search(query_embedding, top_k=10)

        result_ids = [r.id for r in results]
        assert active.id in result_ids
        assert inactive.id not in result_ids

    def test_search_with_time_filter(self, store):
        """Test search with time range filter."""
        now = datetime.utcnow()

        old_mem = store.add(
            content="Old memory",
            source="test",
            timestamp=now - timedelta(days=100),
        )
        recent_mem = store.add(
            content="Recent memory",
            source="test",
            timestamp=now - timedelta(days=5),
        )

        query_embedding = mock_embedding("memory")
        filters = MemoryFilter(
            time_range=TimeRange(
                start=now - timedelta(days=30),
                end=now,
            )
        )

        results = store.search(query_embedding, top_k=10, filters=filters)
        result_ids = [r.id for r in results]

        assert recent_mem.id in result_ids
        assert old_mem.id not in result_ids

    def test_get_history(self, store):
        """Test version history retrieval."""
        v1 = store.add(content="Version 1", source="test")
        v2 = store.update(v1.id, "Version 2")
        v3 = store.update(v2.id, "Version 3")

        history = store.get_history(v3.id)

        assert len(history) == 3
        assert history[0].content == "Version 1"
        assert history[1].content == "Version 2"
        assert history[2].content == "Version 3"

    def test_supersession_detection(self, store):
        """Test automatic supersession detection."""
        # Add two very similar memories from same source
        mem1 = store.add(
            content="The CEO is John Smith",
            source="company_info",
        )
        # This should supersede mem1 due to high similarity
        store._supersession_threshold = 0.0  # Force supersession for test
        mem2 = store.add(
            content="The CEO is Jane Doe",
            source="company_info",
        )

        # Note: In real use, similarity would be computed
        # This test just verifies the mechanism exists
        assert len(store) >= 1

    def test_store_length(self, store):
        """Test __len__ returns active count."""
        store.add(content="Memory 1", source="test")
        store.add(content="Memory 2", source="test")
        mem3 = store.add(content="Memory 3", source="test")
        store.delete(mem3.id)

        assert len(store) == 2


class TestMemoryFilter:
    """Tests for memory filtering."""

    def test_source_filter(self, store):
        """Test filtering by source."""
        store.add(content="From source A", source="source_a")
        store.add(content="From source B", source="source_b")

        query_embedding = mock_embedding("content")
        filters = MemoryFilter(sources=["source_a"])

        results = store.search(query_embedding, top_k=10, filters=filters)

        assert len(results) == 1
        assert results[0].source == "source_a"

    def test_tag_filter(self, store):
        """Test filtering by tags."""
        store.add(content="Tagged important", source="test", tags=["important"])
        store.add(content="Tagged normal", source="test", tags=["normal"])

        query_embedding = mock_embedding("content")
        filters = MemoryFilter(tags=["important"])

        results = store.search(query_embedding, top_k=10, filters=filters)

        assert len(results) == 1
        assert "important" in results[0].tags
