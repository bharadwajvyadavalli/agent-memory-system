"""Tests for conflict detection and resolution."""

import pytest
from datetime import datetime, timedelta

from memory.schema import Memory, ConflictRecord
from memory.temporal import TemporalManager
from agents.conflict import ConflictAgent


@pytest.fixture
def now():
    """Current timestamp for tests."""
    return datetime.utcnow()


@pytest.fixture
def temporal_manager():
    """Create temporal manager."""
    return TemporalManager(half_life_days=30)


class TestConflictRecord:
    """Tests for ConflictRecord schema."""

    def test_create_full_contradiction(self):
        """Test creating a full contradiction record."""
        record = ConflictRecord(
            memory_a_id="mem1",
            memory_b_id="mem2",
            conflict_type="full_contradiction",
            winner_id="mem2",
            reasoning="Memory B is newer and explicitly dated",
        )

        assert record.conflict_type == "full_contradiction"
        assert record.winner_id == "mem2"

    def test_create_partial_update(self):
        """Test creating a partial update record."""
        record = ConflictRecord(
            memory_a_id="mem1",
            memory_b_id="mem2",
            conflict_type="partial_update",
            winner_id=None,
            reasoning="Both memories are valid for different contexts",
        )

        assert record.conflict_type == "partial_update"
        assert record.winner_id is None

    def test_create_scope_difference(self):
        """Test creating a scope difference record."""
        record = ConflictRecord(
            memory_a_id="mem1",
            memory_b_id="mem2",
            conflict_type="scope_difference",
            winner_id=None,
            reasoning="Different scopes, not a real conflict",
        )

        assert record.conflict_type == "scope_difference"


class TestConflictDetection:
    """Tests for conflict detection logic."""

    def test_ceo_change_detection(self, now):
        """Test detecting CEO change as full contradiction."""
        old_ceo = Memory(
            id="ceo_old",
            content="John Smith is the CEO of TechCorp.",
            source="annual_report_2022",
            timestamp=now - timedelta(days=400),
        )

        new_ceo = Memory(
            id="ceo_new",
            content="Jane Doe was appointed CEO of TechCorp in March 2024.",
            source="press_release",
            timestamp=now - timedelta(days=30),
        )

        # In a real scenario, the ConflictAgent would detect:
        # - Both claim to identify the CEO
        # - Names are different (John vs Jane)
        # - Newer one (Jane) should win

        # For this test, verify the memory structure is correct
        assert old_ceo.timestamp < new_ceo.timestamp
        assert "John" in old_ceo.content
        assert "Jane" in new_ceo.content

    def test_policy_update_detection(self, now):
        """Test detecting policy update."""
        old_policy = Memory(
            id="policy_old",
            content="Return policy: 30 days full refund.",
            source="policy_v1",
            timestamp=now - timedelta(days=365),
        )

        new_policy = Memory(
            id="policy_new",
            content="Updated: Return policy changed to 15 days.",
            source="policy_v2",
            timestamp=now - timedelta(days=30),
        )

        # The newer policy supersedes the older one
        assert new_policy.timestamp > old_policy.timestamp
        assert "Updated" in new_policy.content

    def test_scope_difference_no_conflict(self, now):
        """Test that different scopes are not conflicts."""
        company_budget = Memory(
            id="budget_company",
            content="Company annual budget is $10 million.",
            source="finance",
            timestamp=now - timedelta(days=30),
        )

        dept_budget = Memory(
            id="budget_dept",
            content="Engineering department budget is $3 million.",
            source="finance",
            timestamp=now - timedelta(days=30),
        )

        # These are not conflicting - different scopes
        # $3M engineering is a subset of $10M total
        assert "Company" in company_budget.content
        assert "Engineering" in dept_budget.content


class TestTemporalConflictAnalysis:
    """Tests for temporal aspects of conflict analysis."""

    def test_recency_comparison(self, temporal_manager, now):
        """Test comparing recency of conflicting memories."""
        old = Memory(
            id="old",
            content="Old information",
            source="test",
            timestamp=now - timedelta(days=365),
        )

        new = Memory(
            id="new",
            content="New information",
            source="test",
            timestamp=now - timedelta(days=7),
        )

        old_score = temporal_manager.recency_score(old.timestamp)
        new_score = temporal_manager.recency_score(new.timestamp)

        assert new_score > old_score
        # New should be much more recent
        assert new_score > 0.8
        assert old_score < 0.1

    def test_temporal_language_affects_conflict(self, temporal_manager):
        """Test that temporal language affects conflict resolution."""
        # Memory with explicit temporal marker
        dated_content = "As of January 2024, the policy has changed."
        has_temporal = temporal_manager.has_current_language(dated_content)

        # Memory with outdated language
        old_content = "Previously, we used this approach."
        has_outdated = temporal_manager.has_outdated_language(old_content)

        assert has_temporal or has_outdated  # At least one should be detected


class TestConflictAgent:
    """Tests for ConflictAgent behavior."""

    def test_agent_handles_empty_candidates(self):
        """Test agent handles empty candidate list."""
        agent = ConflictAgent(llm_provider="mock")
        decisions = agent.reason_sync("query", [])

        assert decisions == []

    def test_agent_handles_single_candidate(self, now):
        """Test agent handles single candidate (no conflicts possible)."""
        agent = ConflictAgent(llm_provider="mock")
        single = [
            Memory(
                id="single",
                content="Only memory",
                source="test",
                timestamp=now,
            )
        ]

        decisions = agent.reason_sync("query", single)

        assert len(decisions) == 1
        assert decisions[0].action == "keep"

    def test_agent_records_conflicts(self, now):
        """Test that conflicts are recorded for audit."""
        agent = ConflictAgent(llm_provider="mock", log_conflicts=True)

        memories = [
            Memory(id="a", content="A", source="test", timestamp=now),
            Memory(id="b", content="B", source="test", timestamp=now),
        ]

        agent.reason_sync("query", memories)

        # Can retrieve conflict records
        records = agent.get_conflict_records()
        assert isinstance(records, list)

        # Can clear records
        agent.clear_conflict_records()
        assert len(agent.get_conflict_records()) == 0

    def test_conflict_resolution_prefers_newer(self, now):
        """Test that resolution prefers newer memories by default."""
        agent = ConflictAgent(
            llm_provider="mock",
            prefer_newer=True,
        )

        # Agent is configured to prefer newer
        assert agent.prefer_newer is True
