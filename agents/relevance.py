"""RelevanceAgent: Determines contextual relevance beyond embedding similarity."""

import logging
from typing import Optional

from agents.base import BaseMemoryAgent
from memory.schema import AgentDecision, Memory

logger = logging.getLogger(__name__)


# System prompt for relevance evaluation
RELEVANCE_SYSTEM_PROMPT = """You are a relevance evaluation agent in a memory retrieval system. Your job is to determine if candidate memories are actually relevant to the user's query, going beyond simple keyword or embedding similarity.

Your task:
1. Analyze each candidate memory against the query
2. Determine if it's genuinely relevant or just superficially similar
3. Provide a confidence score (0.0 to 1.0) and reasoning

Key considerations:
- "Python performance" should NOT match "Monty Python" - same word, different context
- "Apple stock price" should NOT match "apple pie recipe" - different domains
- "Current CEO" should match memories about leadership, not just ones containing "CEO"
- Consider the user's actual intent, not just word overlap

Output format (JSON):
{
  "decisions": [
    {
      "memory_id": "<id>",
      "action": "keep" | "discard",
      "confidence": 0.0-1.0,
      "reasoning": "Brief explanation"
    }
  ]
}

Few-shot examples:

Query: "Python performance optimization"
Memory: "Monty Python's Flying Circus was a British comedy show from 1969-1974."
Decision: {"memory_id": "...", "action": "discard", "confidence": 0.95, "reasoning": "This is about the comedy show Monty Python, not the Python programming language. The user is asking about code performance."}

Query: "Apple quarterly earnings"
Memory: "Apple released the iPhone 15 with improved camera features."
Decision: {"memory_id": "...", "action": "keep", "confidence": 0.7, "reasoning": "While not directly about earnings, product releases can impact quarterly performance. Moderately relevant context."}

Query: "Best Italian restaurants nearby"
Memory: "Italian is a Romance language spoken by 85 million people."
Decision: {"memory_id": "...", "action": "discard", "confidence": 0.9, "reasoning": "This is about the Italian language, not Italian cuisine or restaurants."}
"""


class RelevanceAgent(BaseMemoryAgent):
    """Agent that evaluates contextual relevance of memories to queries."""

    def __init__(
        self,
        confidence_threshold: float = 0.4,
        max_candidates: int = 50,
        **kwargs,
    ):
        """Initialize the relevance agent.

        Args:
            confidence_threshold: Minimum confidence to keep a memory.
            max_candidates: Maximum candidates to process in one batch.
            **kwargs: Arguments passed to BaseMemoryAgent.
        """
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.max_candidates = max_candidates

    async def reason(
        self,
        query: str,
        candidates: list[Memory],
        context: Optional[dict] = None,
    ) -> list[AgentDecision]:
        """Evaluate relevance of each candidate to the query.

        Args:
            query: The user's query.
            candidates: List of candidate memories from embedding search.
            context: Optional additional context.

        Returns:
            List of AgentDecision objects with relevance judgments.
        """
        if not candidates:
            return []

        # Limit candidates if necessary
        if len(candidates) > self.max_candidates:
            logger.warning(
                f"RelevanceAgent: Truncating {len(candidates)} candidates to {self.max_candidates}"
            )
            candidates = candidates[: self.max_candidates]

        # Format prompt
        user_prompt = self._build_user_prompt(query, candidates)

        try:
            # Call LLM
            response = await self._call_llm(RELEVANCE_SYSTEM_PROMPT, user_prompt)

            # Parse decisions
            decisions = self._parse_decisions(response, candidates)

            # Apply confidence threshold
            for decision in decisions:
                if decision.action == "keep" and decision.confidence < self.confidence_threshold:
                    decision.action = "discard"
                    decision.reasoning += f" (below threshold {self.confidence_threshold})"

            logger.info(
                f"RelevanceAgent: {sum(1 for d in decisions if d.action == 'keep')}/{len(decisions)} kept"
            )

            return decisions

        except Exception as e:
            logger.error(f"RelevanceAgent failed: {e}")
            # Graceful degradation: keep all with medium confidence
            return [
                AgentDecision(
                    agent_name=self.name,
                    memory_id=mem.id,
                    action="keep",
                    confidence=0.5,
                    reasoning=f"Agent error, passing through: {str(e)[:50]}",
                )
                for mem in candidates
            ]

    def _build_user_prompt(self, query: str, candidates: list[Memory]) -> str:
        """Build the user prompt for relevance evaluation.

        Args:
            query: The user's query.
            candidates: Candidate memories.

        Returns:
            Formatted user prompt.
        """
        candidates_text = self._format_candidates_for_prompt(candidates)

        return f"""Query: "{query}"

Candidate memories to evaluate:

{candidates_text}

Evaluate each memory for relevance to the query. For each memory, determine:
1. Is it genuinely relevant to what the user is asking?
2. Or is it just superficially similar (shared words but different context)?

Return your decisions as JSON."""
