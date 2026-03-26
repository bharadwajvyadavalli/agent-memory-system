"""Tests for agent reasoning."""

import pytest
from datetime import datetime, timedelta

from memory.schema import Memory, AgentDecision
from agents.base import BaseMemoryAgent
from agents.relevance import RelevanceAgent
from agents.recency import RecencyAgent
from agents.conflict import ConflictAgent
from agents.synthesis import SynthesisAgent


@pytest.fixture
def mock_memories():
    """Create mock memories for testing."""
    now = datetime.utcnow()
    return [
        Memory(
            id="mem1",
            content="Python 3.12 has performance improvements.",
            source="docs",
            timestamp=now - timedelta(days=30),
            embedding=[0.1] * 384,
            tags=["python"],
        ),
        Memory(
            id="mem2",
            content="Monty Python was a comedy group.",
            source="wikipedia",
            timestamp=now - timedelta(days=100),
            embedding=[0.2] * 384,
            tags=["comedy"],
        ),
        Memory(
            id="mem3",
            content="Python type hints improve code quality.",
            source="tutorial",
            timestamp=now - timedelta(days=60),
            embedding=[0.15] * 384,
            tags=["python"],
        ),
    ]


class TestRelevanceAgent:
    """Tests for RelevanceAgent."""

    def test_init(self):
        """Test agent initialization."""
        agent = RelevanceAgent(llm_provider="mock")
        assert agent.name == "RelevanceAgent"
        assert agent.confidence_threshold == 0.4

    def test_reason_returns_decisions(self, mock_memories):
        """Test that reason returns decisions for all candidates."""
        agent = RelevanceAgent(llm_provider="mock")
        decisions = agent.reason_sync("Python performance", mock_memories)

        assert len(decisions) == len(mock_memories)
        for decision in decisions:
            assert isinstance(decision, AgentDecision)
            assert decision.agent_name == "RelevanceAgent"
            assert decision.action in ["keep", "discard", "merge", "flag_conflict"]

    def test_format_candidates(self, mock_memories):
        """Test candidate formatting for prompts."""
        agent = RelevanceAgent(llm_provider="mock")
        formatted = agent._format_candidates_for_prompt(mock_memories)

        assert "mem1" in formatted
        assert "Python 3.12" in formatted
        assert "docs" in formatted


class TestRecencyAgent:
    """Tests for RecencyAgent."""

    def test_init(self):
        """Test agent initialization."""
        agent = RecencyAgent(llm_provider="mock", half_life_days=30)
        assert agent.name == "RecencyAgent"
        assert agent.temporal_manager.half_life_days == 30

    def test_reason_includes_temporal_analysis(self, mock_memories):
        """Test that recency agent includes temporal analysis."""
        agent = RecencyAgent(llm_provider="mock")
        decisions = agent.reason_sync("query", mock_memories)

        assert len(decisions) == len(mock_memories)

    def test_metadata_analysis(self, mock_memories):
        """Test metadata-based temporal analysis."""
        agent = RecencyAgent(llm_provider="mock")
        analysis = agent._analyze_metadata(mock_memories)

        assert "mem1" in analysis
        assert "recency_score" in analysis["mem1"]
        assert "is_stale" in analysis["mem1"]


class TestConflictAgent:
    """Tests for ConflictAgent."""

    def test_init(self):
        """Test agent initialization."""
        agent = ConflictAgent(llm_provider="mock")
        assert agent.name == "ConflictAgent"
        assert agent.prefer_newer is True

    def test_reason_with_single_memory(self):
        """Test that single memory returns keep decision."""
        agent = ConflictAgent(llm_provider="mock")
        single_memory = [
            Memory(
                id="single",
                content="Only memory",
                source="test",
                timestamp=datetime.utcnow(),
            )
        ]

        decisions = agent.reason_sync("query", single_memory)

        assert len(decisions) == 1
        assert decisions[0].action == "keep"

    def test_conflict_record_storage(self, mock_memories):
        """Test that conflicts are recorded."""
        agent = ConflictAgent(llm_provider="mock", log_conflicts=True)
        agent.reason_sync("query", mock_memories)

        # With mock provider, no actual conflicts detected
        records = agent.get_conflict_records()
        assert isinstance(records, list)


class TestSynthesisAgent:
    """Tests for SynthesisAgent."""

    def test_init(self):
        """Test agent initialization."""
        agent = SynthesisAgent(llm_provider="mock", max_tokens=2048)
        assert agent.name == "SynthesisAgent"
        assert agent.max_tokens == 2048

    def test_reason_stores_synthesis(self, mock_memories):
        """Test that synthesis is stored after reasoning."""
        agent = SynthesisAgent(llm_provider="mock")
        agent.reason_sync("Python features", mock_memories)

        synthesis = agent.get_last_synthesis()
        assert synthesis is not None

    def test_token_estimation(self):
        """Test token estimation."""
        agent = SynthesisAgent(llm_provider="mock")

        text = "This is a test sentence with ten words in it."
        tokens = agent._estimate_tokens(text)

        # Rough estimate: ~10 words / 0.75 ≈ 13 tokens
        assert 10 <= tokens <= 20

    def test_fallback_synthesis(self, mock_memories):
        """Test fallback synthesis when LLM fails."""
        agent = SynthesisAgent(llm_provider="mock")
        synthesis = agent._fallback_synthesis(mock_memories)

        assert len(synthesis) > 0
        for mem in mock_memories:
            assert mem.content in synthesis or mem.content[:20] in synthesis


class TestBaseMemoryAgent:
    """Tests for BaseMemoryAgent functionality."""

    def test_parse_decisions_with_valid_json(self, mock_memories):
        """Test parsing valid JSON response."""
        agent = RelevanceAgent(llm_provider="mock")

        response = '''```json
{
    "decisions": [
        {"memory_id": "mem1", "action": "keep", "confidence": 0.9, "reasoning": "Relevant"},
        {"memory_id": "mem2", "action": "discard", "confidence": 0.95, "reasoning": "Irrelevant"}
    ]
}
```'''

        decisions = agent._parse_decisions(response, mock_memories)

        assert len(decisions) == len(mock_memories)
        mem1_decision = next(d for d in decisions if d.memory_id == "mem1")
        assert mem1_decision.action == "keep"
        assert mem1_decision.confidence == 0.9

    def test_parse_decisions_with_invalid_json(self, mock_memories):
        """Test graceful fallback with invalid JSON."""
        agent = RelevanceAgent(llm_provider="mock")

        invalid_response = "This is not valid JSON"
        decisions = agent._parse_decisions(invalid_response, mock_memories)

        # Should fall back to keep all with low confidence
        assert len(decisions) == len(mock_memories)
        for decision in decisions:
            assert decision.action == "keep"
            assert decision.confidence == 0.3
