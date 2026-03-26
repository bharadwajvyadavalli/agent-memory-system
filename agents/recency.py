"""RecencyAgent: Temporal reasoning, staleness filtering, and recency scoring."""

import logging
from typing import Optional

from agents.base import BaseMemoryAgent
from memory.schema import AgentDecision, Memory
from memory.temporal import TemporalManager

logger = logging.getLogger(__name__)


RECENCY_SYSTEM_PROMPT = """You are a temporal reasoning agent in a memory retrieval system. Your job is to evaluate memories for recency and staleness, considering both timestamps and temporal language in the content.

Your tasks:
1. Assess if each memory is current or potentially stale
2. Detect temporal language indicating currency ("current", "as of 2024") or outdated status ("was", "previously")
3. Identify if newer information exists that supersedes older memories

Key considerations:
- Timestamps matter: a policy from 2022 is likely outdated if there's a 2024 update
- Temporal language matters: "as of 2023" signals a point-in-time snapshot
- Source type matters: news is stale in days, policies in months, facts in years
- "Current CEO" claims from old documents may no longer be valid

Output format (JSON):
{
  "decisions": [
    {
      "memory_id": "<id>",
      "action": "keep" | "discard" | "flag_conflict",
      "confidence": 0.0-1.0,
      "reasoning": "Brief explanation",
      "recency_score": 0.0-1.0,
      "temporal_signals": ["current", "outdated", "versioned", "none"]
    }
  ]
}

Actions:
- keep: Memory is current and valid
- discard: Memory is stale or superseded
- flag_conflict: Newer version exists but doesn't fully supersede (partial update)

Examples:

Memory (2022-01): "Our return policy is 30 days no questions asked."
Memory (2024-01): "Updated: Return policy changed to 15 days effective January 2024."
For query "What's the return policy?":
- 2022 memory: discard (superseded by 2024 update)
- 2024 memory: keep (most recent, contains temporal marker)

Memory (2023-06): "As of Q2 2023, our user count reached 1 million."
For query "How many users?":
- Action: keep with note that this is a snapshot, not current
- Temporal signal: "versioned" (specific point in time)
"""


class RecencyAgent(BaseMemoryAgent):
    """Agent that performs temporal reasoning on memories."""

    def __init__(
        self,
        half_life_days: float = 30.0,
        staleness_thresholds: Optional[dict[str, int]] = None,
        cross_reference_enabled: bool = True,
        **kwargs,
    ):
        """Initialize the recency agent.

        Args:
            half_life_days: Half-life for recency decay scoring.
            staleness_thresholds: Custom staleness thresholds by source type.
            cross_reference_enabled: Whether to cross-reference memories for supersession.
            **kwargs: Arguments passed to BaseMemoryAgent.
        """
        super().__init__(**kwargs)
        self.temporal_manager = TemporalManager(
            half_life_days=half_life_days,
            staleness_thresholds=staleness_thresholds,
        )
        self.cross_reference_enabled = cross_reference_enabled

    async def reason(
        self,
        query: str,
        candidates: list[Memory],
        context: Optional[dict] = None,
    ) -> list[AgentDecision]:
        """Evaluate temporal aspects of candidate memories.

        Args:
            query: The user's query.
            candidates: List of candidate memories.
            context: Optional additional context.

        Returns:
            List of AgentDecision objects with temporal assessments.
        """
        if not candidates:
            return []

        # Phase 1: Metadata-based temporal analysis
        metadata_analysis = self._analyze_metadata(candidates)

        # Phase 2: Content-based temporal analysis via LLM
        user_prompt = self._build_user_prompt(query, candidates, metadata_analysis)

        try:
            response = await self._call_llm(RECENCY_SYSTEM_PROMPT, user_prompt)
            decisions = self._parse_decisions(response, candidates)

            # Merge metadata analysis with LLM decisions
            for decision in decisions:
                mem_id = decision.memory_id
                if mem_id in metadata_analysis:
                    meta = metadata_analysis[mem_id]
                    # If metadata shows very stale, override to discard
                    if meta["is_stale"] and decision.action == "keep":
                        decision.action = "discard"
                        decision.reasoning += " (metadata: stale based on source type)"
                        decision.confidence = max(decision.confidence, meta["recency_score"])

            logger.info(
                f"RecencyAgent: {sum(1 for d in decisions if d.action == 'keep')}/{len(decisions)} kept"
            )

            return decisions

        except Exception as e:
            logger.error(f"RecencyAgent failed: {e}")
            # Graceful degradation: use metadata analysis only
            return [
                AgentDecision(
                    agent_name=self.name,
                    memory_id=mem.id,
                    action="keep" if not metadata_analysis.get(mem.id, {}).get("is_stale", False) else "discard",
                    confidence=metadata_analysis.get(mem.id, {}).get("recency_score", 0.5),
                    reasoning=f"LLM failed, using metadata: recency={metadata_analysis.get(mem.id, {}).get('recency_score', 0.5):.2f}",
                )
                for mem in candidates
            ]

    def _analyze_metadata(self, candidates: list[Memory]) -> dict[str, dict]:
        """Perform metadata-based temporal analysis.

        Args:
            candidates: List of memories to analyze.

        Returns:
            Dict mapping memory_id to analysis results.
        """
        analysis = {}

        for mem in candidates:
            recency_score = self.temporal_manager.recency_score(mem.timestamp)
            is_stale = self.temporal_manager.is_stale(mem)
            has_current_lang = self.temporal_manager.has_current_language(mem.content)
            has_outdated_lang = self.temporal_manager.has_outdated_language(mem.content)

            analysis[mem.id] = {
                "recency_score": recency_score,
                "is_stale": is_stale,
                "has_current_language": has_current_lang,
                "has_outdated_language": has_outdated_lang,
                "timestamp": mem.timestamp.isoformat(),
            }

        return analysis

    def _build_user_prompt(
        self,
        query: str,
        candidates: list[Memory],
        metadata_analysis: dict[str, dict],
    ) -> str:
        """Build the user prompt for temporal reasoning.

        Args:
            query: The user's query.
            candidates: Candidate memories.
            metadata_analysis: Results from metadata analysis.

        Returns:
            Formatted user prompt.
        """
        # Include metadata analysis in the prompt
        candidates_with_meta = []
        for mem in candidates:
            meta = metadata_analysis.get(mem.id, {})
            candidates_with_meta.append(
                f"Memory (ID: {mem.id}):\n"
                f"  Content: {mem.content}\n"
                f"  Source: {mem.source}\n"
                f"  Timestamp: {mem.timestamp.isoformat()}\n"
                f"  Recency Score: {meta.get('recency_score', 0):.2f}\n"
                f"  Metadata Stale: {meta.get('is_stale', False)}\n"
                f"  Has Current Language: {meta.get('has_current_language', False)}\n"
                f"  Has Outdated Language: {meta.get('has_outdated_language', False)}"
            )

        candidates_text = "\n\n".join(candidates_with_meta)

        # Check for temporal context in query
        temporal_context = self.temporal_manager.get_temporal_context(query)
        temporal_note = ""
        if temporal_context:
            temporal_note = f"\nNote: Query has temporal context suggesting interest in period {temporal_context[0]} to {temporal_context[1]}."

        return f"""Query: "{query}"{temporal_note}

Candidate memories to evaluate for temporal validity:

{candidates_text}

For each memory:
1. Is it current or potentially stale?
2. Does it contain temporal language indicating when it was valid?
3. Is there a newer memory that supersedes it?

Return your decisions as JSON."""
