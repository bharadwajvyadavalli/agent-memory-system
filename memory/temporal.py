"""Temporal reasoning utilities for memory recency and staleness."""

import re
from datetime import datetime, timedelta
from typing import Optional

from memory.schema import Memory


class TemporalManager:
    """Manages temporal aspects of memories: recency scoring, staleness detection."""

    # Default staleness thresholds by source type (in days)
    DEFAULT_STALENESS_THRESHOLDS = {
        "news": 7,
        "policy": 90,
        "documentation": 180,
        "fact": 365,
        "default": 180,
    }

    # Temporal reference patterns
    TEMPORAL_PATTERNS = [
        # Explicit dates
        (r"\b(as of|since|from|until)\s+(\d{4})", "year_reference"),
        (r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b", "full_date"),
        (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "date_slash"),
        (r"\b\d{4}-\d{2}-\d{2}\b", "date_iso"),
        # Relative references
        (r"\b(current|currently|now|today|present)\b", "current"),
        (r"\b(latest|newest|most recent|updated)\b", "latest"),
        (r"\b(previous|former|old|outdated|deprecated|legacy)\b", "outdated"),
        (r"\b(last|past)\s+(week|month|year|quarter)\b", "relative_past"),
        (r"\b(this|next)\s+(week|month|year|quarter)\b", "relative_current"),
        # Version indicators
        (r"\bv\d+(\.\d+)*\b", "version"),
        (r"\b(version|release)\s+\d+(\.\d+)*\b", "version_explicit"),
        # Temporal language
        (r"\b(effective immediately|starting today|begins now)\b", "immediate"),
        (r"\b(was|were|used to be|previously)\b", "past_tense"),
        (r"\b(will be|upcoming|planned|future)\b", "future"),
    ]

    def __init__(
        self,
        half_life_days: float = 30.0,
        staleness_thresholds: Optional[dict[str, int]] = None,
    ):
        """Initialize the temporal manager.

        Args:
            half_life_days: Half-life for exponential decay in days.
            staleness_thresholds: Custom staleness thresholds by source type.
        """
        self.half_life_days = half_life_days
        self.staleness_thresholds = {
            **self.DEFAULT_STALENESS_THRESHOLDS,
            **(staleness_thresholds or {}),
        }
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), label)
            for pattern, label in self.TEMPORAL_PATTERNS
        ]

    def recency_score(
        self,
        timestamp: datetime,
        reference_time: Optional[datetime] = None,
        half_life_days: Optional[float] = None,
    ) -> float:
        """Calculate recency score using exponential decay.

        Score = 0.5^(age_days / half_life_days)

        Args:
            timestamp: When the memory was created.
            reference_time: Reference time for age calculation (defaults to now).
            half_life_days: Override default half-life.

        Returns:
            Recency score between 0 and 1 (1 = most recent).
        """
        ref = reference_time or datetime.utcnow()
        half_life = half_life_days or self.half_life_days

        age_days = (ref - timestamp).total_seconds() / 86400  # seconds in a day

        if age_days <= 0:
            return 1.0

        return 0.5 ** (age_days / half_life)

    def is_stale(
        self,
        memory: Memory,
        reference_time: Optional[datetime] = None,
    ) -> bool:
        """Check if a memory is stale based on source type.

        Args:
            memory: The memory to check.
            reference_time: Reference time for staleness check.

        Returns:
            True if the memory is considered stale.
        """
        ref = reference_time or datetime.utcnow()

        # Determine staleness threshold based on source
        source_type = self._infer_source_type(memory.source)
        threshold_days = self.staleness_thresholds.get(
            source_type, self.staleness_thresholds["default"]
        )

        age_days = (ref - memory.timestamp).total_seconds() / 86400

        return age_days > threshold_days

    def get_temporal_context(self, query: str) -> Optional[tuple[datetime, datetime]]:
        """Extract temporal context from a query.

        Args:
            query: The search query.

        Returns:
            Optional (start, end) datetime tuple, or None if no temporal context found.
        """
        query_lower = query.lower()

        # Check for year references
        year_match = re.search(r"\b(20\d{2})\b", query)
        if year_match:
            year = int(year_match.group(1))
            start = datetime(year, 1, 1)
            end = datetime(year, 12, 31, 23, 59, 59)
            return (start, end)

        # Check for relative time references
        if any(word in query_lower for word in ["current", "now", "today", "latest"]):
            # Bias towards recent memories
            end = datetime.utcnow()
            start = end - timedelta(days=30)  # Last 30 days
            return (start, end)

        if "last week" in query_lower:
            end = datetime.utcnow()
            start = end - timedelta(days=7)
            return (start, end)

        if "last month" in query_lower:
            end = datetime.utcnow()
            start = end - timedelta(days=30)
            return (start, end)

        if "last year" in query_lower:
            end = datetime.utcnow()
            start = end - timedelta(days=365)
            return (start, end)

        return None

    def decay_weights(
        self,
        memories: list[Memory],
        half_life_days: Optional[float] = None,
        reference_time: Optional[datetime] = None,
    ) -> list[tuple[Memory, float]]:
        """Apply recency scoring to a list of memories.

        Args:
            memories: List of memories to score.
            half_life_days: Override default half-life.
            reference_time: Reference time for scoring.

        Returns:
            List of (Memory, recency_score) tuples, sorted by score descending.
        """
        scored = [
            (memory, self.recency_score(memory.timestamp, reference_time, half_life_days))
            for memory in memories
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def detect_temporal_language(self, content: str) -> list[dict]:
        """Detect temporal language in content.

        Args:
            content: Text content to analyze.

        Returns:
            List of detected temporal references with type and match.
        """
        results = []
        for pattern, label in self._compiled_patterns:
            matches = pattern.findall(content)
            if matches:
                results.append({
                    "type": label,
                    "matches": matches,
                    "indicates_current": label in ["current", "latest", "immediate"],
                    "indicates_outdated": label in ["outdated", "past_tense"],
                })
        return results

    def has_outdated_language(self, content: str) -> bool:
        """Check if content contains language indicating it may be outdated.

        Args:
            content: Text to check.

        Returns:
            True if outdated language detected.
        """
        detections = self.detect_temporal_language(content)
        return any(d.get("indicates_outdated", False) for d in detections)

    def has_current_language(self, content: str) -> bool:
        """Check if content contains language indicating it is current.

        Args:
            content: Text to check.

        Returns:
            True if current/latest language detected.
        """
        detections = self.detect_temporal_language(content)
        return any(d.get("indicates_current", False) for d in detections)

    def _infer_source_type(self, source: str) -> str:
        """Infer source type from source identifier.

        Args:
            source: Source identifier string.

        Returns:
            Source type key for staleness threshold lookup.
        """
        source_lower = source.lower()

        if any(word in source_lower for word in ["news", "article", "blog", "post"]):
            return "news"

        if any(word in source_lower for word in ["policy", "rule", "guideline", "procedure"]):
            return "policy"

        if any(word in source_lower for word in ["doc", "readme", "manual", "guide", "tutorial"]):
            return "documentation"

        if any(word in source_lower for word in ["fact", "data", "stats", "reference"]):
            return "fact"

        return "default"

    def compare_recency(
        self,
        memory_a: Memory,
        memory_b: Memory,
    ) -> int:
        """Compare two memories by recency.

        Args:
            memory_a: First memory.
            memory_b: Second memory.

        Returns:
            -1 if a is older, 0 if same, 1 if a is newer.
        """
        if memory_a.timestamp < memory_b.timestamp:
            return -1
        elif memory_a.timestamp > memory_b.timestamp:
            return 1
        return 0
