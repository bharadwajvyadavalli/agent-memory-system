"""Tests for retrieval pipeline."""

import pytest
from datetime import datetime, timedelta

from memory.schema import Memory, MemoryQuery, RetrievalResult
from memory.store import MemoryStore
from memory.indexer import FAISSIndexer
from retrieval.pipeline import RetrievalPipeline
from retrieval.candidate import CandidateGenerator
from retrieval.context_builder import ContextBuilder
from agents.orchestrator import Orchestrator


def mock_embedding(text: str) -> list[float]:
    """Generate deterministic mock embedding."""
    import hashlib
    h = hashlib.md5(text.encode()).hexdigest()
    return [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)] * 24


@pytest.fixture
def pipeline():
    """Create a pipeline with mock embedding."""
    return RetrievalPipeline(embedding_fn=mock_embedding)


class TestRetrievalPipeline:
    """Tests for RetrievalPipeline."""

    def test_init(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.store is not None
        assert pipeline.orchestrator is not None

    def test_add_memory(self, pipeline):
        """Test adding memory through pipeline."""
        mem = pipeline.add_memory(
            content="Test content",
            source="test",
            tags=["test"],
        )

        assert mem.id is not None
        assert len(pipeline) == 1

    def test_retrieve_fast(self, pipeline):
        """Test fast retrieval without reasoning."""
        pipeline.add_memory("Python programming", source="docs")
        pipeline.add_memory("Machine learning", source="docs")

        result = pipeline.retrieve_fast("Python", top_k=2)

        assert isinstance(result, RetrievalResult)
        assert len(result.agent_decisions) == 0  # No agent reasoning
        assert result.metadata.get("mode") == "fast"

    def test_retrieve_with_reasoning(self, pipeline):
        """Test retrieval with agent reasoning."""
        pipeline.add_memory("Python programming basics", source="docs")
        pipeline.add_memory("JavaScript tutorial", source="docs")

        result = pipeline.retrieve("Python", top_k=2, require_reasoning=True)

        assert isinstance(result, RetrievalResult)
        # With mock agents, all should pass through

    def test_update_memory(self, pipeline):
        """Test memory update through pipeline."""
        original = pipeline.add_memory("Version 1", source="test")
        updated = pipeline.update_memory(original.id, "Version 2")

        assert updated.version == 2
        assert updated.supersedes == original.id

    def test_delete_memory(self, pipeline):
        """Test memory deletion."""
        mem = pipeline.add_memory("To delete", source="test")
        initial_count = len(pipeline)

        result = pipeline.delete_memory(mem.id)

        assert result is True
        assert len(pipeline) == initial_count - 1

    def test_get_memory_history(self, pipeline):
        """Test getting memory history."""
        v1 = pipeline.add_memory("Version 1", source="test")
        v2 = pipeline.update_memory(v1.id, "Version 2")

        history = pipeline.get_memory_history(v2.id)

        assert len(history) == 2

    def test_list_memories(self, pipeline):
        """Test listing all memories."""
        pipeline.add_memory("Memory 1", source="test")
        pipeline.add_memory("Memory 2", source="test")

        memories = pipeline.list_memories()

        assert len(memories) == 2


class TestCandidateGenerator:
    """Tests for CandidateGenerator."""

    @pytest.fixture
    def generator(self):
        """Create candidate generator."""
        store = MemoryStore(
            indexer=FAISSIndexer(dimension=384),
            embedding_fn=mock_embedding,
        )
        return CandidateGenerator(store=store, embedding_fn=mock_embedding)

    def test_generate(self, generator):
        """Test candidate generation."""
        # Add memories to store
        generator._store.add("Python basics", source="docs")
        generator._store.add("JavaScript intro", source="docs")

        candidates = generator.generate("Python", top_k=5)

        assert isinstance(candidates, list)

    def test_generate_with_filters(self, generator):
        """Test candidate generation with filters."""
        generator._store.add("Python basics", source="docs_python")
        generator._store.add("JavaScript intro", source="docs_js")

        candidates = generator.generate(
            "programming",
            top_k=5,
            filters={"sources": ["docs_python"]},
        )

        for c in candidates:
            assert c.source == "docs_python"


class TestContextBuilder:
    """Tests for ContextBuilder."""

    @pytest.fixture
    def builder(self):
        """Create context builder."""
        return ContextBuilder(
            max_tokens=500,
            include_timestamps=True,
            include_source=True,
        )

    @pytest.fixture
    def sample_memories(self):
        """Create sample memories."""
        return [
            Memory(
                id="mem1",
                content="First memory content",
                source="source1",
                timestamp=datetime.utcnow(),
            ),
            Memory(
                id="mem2",
                content="Second memory content",
                source="source2",
                timestamp=datetime.utcnow() - timedelta(days=1),
            ),
        ]

    def test_build_basic(self, builder, sample_memories):
        """Test basic context building."""
        context = builder.build(sample_memories)

        assert "First memory" in context
        assert "Second memory" in context
        assert "[Source:" in context

    def test_build_with_synthesis(self, builder, sample_memories):
        """Test building with pre-built synthesis."""
        synthesis = "Pre-built synthesis content"
        context = builder.build(sample_memories, synthesis=synthesis)

        assert context == synthesis

    def test_build_minimal(self, builder, sample_memories):
        """Test minimal context (content only)."""
        context = builder.build_minimal(sample_memories)

        assert "First memory content" in context
        assert "[Source:" not in context

    def test_build_structured(self, builder, sample_memories):
        """Test structured context output."""
        result = builder.build_structured(sample_memories)

        assert "memories" in result
        assert "count" in result
        assert "text" in result
        assert result["count"] == 2

    def test_token_budgeting(self, builder, sample_memories):
        """Test that context respects token budget."""
        builder.max_tokens = 10  # Very small budget

        context = builder.build(sample_memories)

        # Should be truncated
        estimated_tokens = builder._estimate_tokens(context)
        # Allow some slack for truncation marker
        assert estimated_tokens <= 15


class TestOrchestrator:
    """Tests for Orchestrator."""

    def test_init_default_agents(self):
        """Test orchestrator initializes default agents."""
        orchestrator = Orchestrator()

        assert "relevance" in orchestrator.agents
        assert "recency" in orchestrator.agents
        assert "conflict" in orchestrator.agents
        assert "synthesis" in orchestrator.agents

    def test_configure(self):
        """Test orchestrator configuration."""
        orchestrator = Orchestrator()

        orchestrator.configure(
            agent_order=["relevance", "synthesis"],
            skip_agents=["recency", "conflict"],
        )

        assert orchestrator.agent_order == ["relevance", "synthesis"]
        assert "recency" in orchestrator.skip_agents

    def test_run_sync(self):
        """Test synchronous run."""
        orchestrator = Orchestrator()
        query = MemoryQuery(query="test", require_reasoning=False)

        result = orchestrator.run_sync(query, [])

        assert isinstance(result, RetrievalResult)
