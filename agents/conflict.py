"""ConflictAgent: Contradiction detection and supersession resolution."""

import logging
from itertools import combinations
from typing import Optional

from agents.base import BaseMemoryAgent
from memory.schema import AgentDecision, ConflictRecord, Memory

logger = logging.getLogger(__name__)


CONFLICT_SYSTEM_PROMPT = """You are a conflict detection agent in a memory retrieval system. Your job is to identify contradictions between memories and determine which should be preferred.

Conflict types:
1. full_contradiction: Memories directly contradict (e.g., "CEO is John" vs "CEO is Jane"). Newer typically wins.
2. partial_update: One memory updates part of another (e.g., "policy is X" vs "policy changed, now Y for some cases")
3. scope_difference: Memories have different scopes (e.g., "company budget is $1M" vs "engineering budget is $400K" - not a conflict)

Your tasks:
1. Compare memories pairwise for potential conflicts
2. Determine conflict type (or "none" if no conflict)
3. Decide which memory should be preferred (winner)
4. Provide reasoning

Output format (JSON):
{
  "conflicts": [
    {
      "memory_a_id": "<id>",
      "memory_b_id": "<id>",
      "conflict_type": "full_contradiction" | "partial_update" | "scope_difference" | "none",
      "winner_id": "<id>" | null,
      "reasoning": "Explanation of conflict and resolution"
    }
  ],
  "decisions": [
    {
      "memory_id": "<id>",
      "action": "keep" | "discard" | "flag_conflict",
      "confidence": 0.0-1.0,
      "reasoning": "Why this action"
    }
  ]
}

Resolution rules:
- full_contradiction: Newer memory wins (check timestamps)
- partial_update: Both may be valid for different contexts, flag_conflict
- scope_difference: Both are valid, keep both

Examples:

Memory A (2023-01): "John Smith is the CEO of TechCorp."
Memory B (2024-03): "Jane Doe was appointed CEO of TechCorp in March 2024."
Conflict: full_contradiction
Winner: Memory B (newer, explicitly dated)
A: discard, B: keep

Memory A: "Our standard warranty is 2 years."
Memory B: "Enterprise customers receive extended 5-year warranty."
Conflict: scope_difference (not a conflict - different customer segments)
Both: keep
"""


class ConflictAgent(BaseMemoryAgent):
    """Agent that detects and resolves contradictions between memories."""

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        prefer_newer: bool = True,
        prefer_specific: bool = True,
        log_conflicts: bool = True,
        **kwargs,
    ):
        """Initialize the conflict agent.

        Args:
            similarity_threshold: Minimum similarity to consider for conflict check.
            prefer_newer: Prefer newer memories in conflicts.
            prefer_specific: Prefer more specific memories.
            log_conflicts: Log detected conflicts.
            **kwargs: Arguments passed to BaseMemoryAgent.
        """
        super().__init__(**kwargs)
        self.similarity_threshold = similarity_threshold
        self.prefer_newer = prefer_newer
        self.prefer_specific = prefer_specific
        self.log_conflicts = log_conflicts
        self.conflict_records: list[ConflictRecord] = []

    async def reason(
        self,
        query: str,
        candidates: list[Memory],
        context: Optional[dict] = None,
    ) -> list[AgentDecision]:
        """Detect conflicts between candidate memories.

        Args:
            query: The user's query.
            candidates: List of candidate memories.
            context: Optional additional context.

        Returns:
            List of AgentDecision objects with conflict resolution.
        """
        if not candidates or len(candidates) < 2:
            # No conflicts possible with 0 or 1 candidates
            return [
                AgentDecision(
                    agent_name=self.name,
                    memory_id=mem.id,
                    action="keep",
                    confidence=0.9,
                    reasoning="No conflicts detected (single memory or empty)",
                )
                for mem in candidates
            ]

        # Generate pairs to check
        pairs = list(combinations(candidates, 2))

        # Build prompt with pairs
        user_prompt = self._build_user_prompt(query, candidates, pairs)

        try:
            response = await self._call_llm(CONFLICT_SYSTEM_PROMPT, user_prompt)
            decisions, conflicts = self._parse_conflict_response(response, candidates)

            # Store conflict records
            if self.log_conflicts and conflicts:
                self.conflict_records.extend(conflicts)
                logger.info(f"ConflictAgent: Detected {len(conflicts)} conflicts")

            logger.info(
                f"ConflictAgent: {sum(1 for d in decisions if d.action == 'keep')}/{len(decisions)} kept"
            )

            return decisions

        except Exception as e:
            logger.error(f"ConflictAgent failed: {e}")
            # Graceful degradation: keep all
            return [
                AgentDecision(
                    agent_name=self.name,
                    memory_id=mem.id,
                    action="keep",
                    confidence=0.5,
                    reasoning=f"Agent error, no conflict detection: {str(e)[:50]}",
                )
                for mem in candidates
            ]

    def _build_user_prompt(
        self,
        query: str,
        candidates: list[Memory],
        pairs: list[tuple[Memory, Memory]],
    ) -> str:
        """Build the user prompt for conflict detection.

        Args:
            query: The user's query.
            candidates: All candidate memories.
            pairs: Pairs of memories to compare.

        Returns:
            Formatted user prompt.
        """
        candidates_text = self._format_candidates_for_prompt(candidates)

        pairs_text = []
        for i, (a, b) in enumerate(pairs[:20], 1):  # Limit pairs to avoid token overflow
            pairs_text.append(f"Pair {i}: Memory {a.id} vs Memory {b.id}")

        return f"""Query: "{query}"

Memories to check for conflicts:

{candidates_text}

Pairs to compare:
{chr(10).join(pairs_text)}

For each pair, determine:
1. Is there a conflict?
2. If yes, what type?
3. Which memory should be preferred?

Then provide final decisions for each memory (keep/discard/flag_conflict).

Return your analysis as JSON."""

    def _parse_conflict_response(
        self,
        llm_response: str,
        candidates: list[Memory],
    ) -> tuple[list[AgentDecision], list[ConflictRecord]]:
        """Parse LLM response with conflict information.

        Args:
            llm_response: Raw LLM response.
            candidates: Original candidates.

        Returns:
            Tuple of (decisions, conflict_records).
        """
        import json

        conflicts = []
        decisions = []

        try:
            # Extract JSON
            response_text = llm_response.strip()
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            data = json.loads(response_text)

            # Parse conflicts
            for c in data.get("conflicts", []):
                if c.get("conflict_type") != "none":
                    conflicts.append(ConflictRecord(
                        memory_a_id=c["memory_a_id"],
                        memory_b_id=c["memory_b_id"],
                        conflict_type=c["conflict_type"],
                        winner_id=c.get("winner_id"),
                        reasoning=c.get("reasoning", ""),
                    ))

            # Parse decisions
            decision_map = {d["memory_id"]: d for d in data.get("decisions", [])}

            for mem in candidates:
                if mem.id in decision_map:
                    d = decision_map[mem.id]
                    decisions.append(AgentDecision(
                        agent_name=self.name,
                        memory_id=mem.id,
                        action=d.get("action", "keep"),
                        confidence=float(d.get("confidence", 0.7)),
                        reasoning=d.get("reasoning", "No reasoning provided"),
                    ))
                else:
                    # Default: keep with medium confidence
                    decisions.append(AgentDecision(
                        agent_name=self.name,
                        memory_id=mem.id,
                        action="keep",
                        confidence=0.6,
                        reasoning="Not explicitly evaluated in conflict analysis",
                    ))

            return decisions, conflicts

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"ConflictAgent parse error: {e}")
            # Fallback: keep all
            decisions = [
                AgentDecision(
                    agent_name=self.name,
                    memory_id=mem.id,
                    action="keep",
                    confidence=0.5,
                    reasoning=f"Parse error, keeping: {str(e)[:50]}",
                )
                for mem in candidates
            ]
            return decisions, []

    def get_conflict_records(self) -> list[ConflictRecord]:
        """Get all recorded conflicts.

        Returns:
            List of ConflictRecord objects.
        """
        return self.conflict_records.copy()

    def clear_conflict_records(self) -> None:
        """Clear recorded conflicts."""
        self.conflict_records.clear()
