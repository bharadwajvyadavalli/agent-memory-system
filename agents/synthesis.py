"""SynthesisAgent: Merge related memories into coherent context."""

import logging
from typing import Optional

from agents.base import BaseMemoryAgent
from memory.schema import AgentDecision, Memory

logger = logging.getLogger(__name__)


SYNTHESIS_SYSTEM_PROMPT = """You are a synthesis agent in a memory retrieval system. Your job is to take a set of relevant, non-conflicting memories and synthesize them into a coherent context for answering the user's query.

Your tasks:
1. Group related memories by topic or entity
2. Synthesize each group into a coherent paragraph
3. Include source attribution for key facts
4. Respect the token budget - summarize if needed
5. Prioritize high-confidence memories

Output format (JSON):
{
  "groups": [
    {
      "topic": "Topic name",
      "memory_ids": ["id1", "id2"],
      "synthesis": "Coherent paragraph combining information from these memories. [Source: source1] [Source: source2]"
    }
  ],
  "final_context": "Complete synthesized context combining all groups.",
  "decisions": [
    {
      "memory_id": "<id>",
      "action": "keep" | "merge",
      "confidence": 0.0-1.0,
      "reasoning": "How this memory was used"
    }
  ],
  "metadata": {
    "groups_created": 2,
    "estimated_tokens": 500
  }
}

Guidelines:
- Maintain factual accuracy - don't invent information
- Use natural language, not bullet points
- Include timestamps where relevant (e.g., "As of March 2024...")
- Attribution format: [Source: source_name]
- If memories provide conflicting confidence levels, favor higher confidence
- Keep the synthesis focused on answering the query

Example:

Query: "What is our company's remote work policy?"

Memory 1: "Effective January 2024, employees can work remotely up to 3 days per week."
Memory 2: "Remote work requires manager approval and a home office setup verification."
Memory 3: "IT provides $500 stipend for home office equipment."

Synthesis:
{
  "groups": [
    {
      "topic": "Remote Work Policy",
      "memory_ids": ["1", "2", "3"],
      "synthesis": "As of January 2024, employees can work remotely up to 3 days per week, subject to manager approval. A home office setup verification is required before beginning remote work. [Source: hr_policy] IT provides a $500 stipend for home office equipment to support remote workers. [Source: it_benefits]"
    }
  ],
  "final_context": "Remote Work Policy: As of January 2024, employees can work remotely up to 3 days per week, subject to manager approval. A home office setup verification is required before beginning remote work. IT provides a $500 stipend for home office equipment to support remote workers.",
  "decisions": [...]
}
"""


class SynthesisAgent(BaseMemoryAgent):
    """Agent that synthesizes multiple memories into coherent context."""

    def __init__(
        self,
        max_tokens: int = 2048,
        grouping_strategy: str = "topic",
        include_attribution: bool = True,
        include_timestamps: bool = True,
        summarization_level: float = 0.5,
        **kwargs,
    ):
        """Initialize the synthesis agent.

        Args:
            max_tokens: Maximum tokens in synthesized context.
            grouping_strategy: How to group memories ("topic", "entity", "source", "none").
            include_attribution: Include source attribution.
            include_timestamps: Include timestamps in synthesis.
            summarization_level: How aggressively to summarize (0.0=verbose, 1.0=minimal).
            **kwargs: Arguments passed to BaseMemoryAgent.
        """
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.grouping_strategy = grouping_strategy
        self.include_attribution = include_attribution
        self.include_timestamps = include_timestamps
        self.summarization_level = summarization_level
        self._last_synthesis: Optional[str] = None

    async def reason(
        self,
        query: str,
        candidates: list[Memory],
        context: Optional[dict] = None,
    ) -> list[AgentDecision]:
        """Synthesize memories into coherent context.

        Args:
            query: The user's query.
            candidates: List of memories to synthesize.
            context: Optional additional context (e.g., decisions from prior agents).

        Returns:
            List of AgentDecision objects indicating how memories were used.
        """
        if not candidates:
            self._last_synthesis = ""
            return []

        # Get confidence scores from context if available
        confidence_map = {}
        if context and "decisions" in context:
            for dec in context["decisions"]:
                confidence_map[dec.memory_id] = dec.confidence

        # Build prompt
        user_prompt = self._build_user_prompt(query, candidates, confidence_map)

        try:
            response = await self._call_llm(SYNTHESIS_SYSTEM_PROMPT, user_prompt)
            decisions, synthesis = self._parse_synthesis_response(response, candidates)

            self._last_synthesis = synthesis

            logger.info(
                f"SynthesisAgent: Synthesized {len(candidates)} memories into ~{self._estimate_tokens(synthesis)} tokens"
            )

            return decisions

        except Exception as e:
            logger.error(f"SynthesisAgent failed: {e}")
            # Fallback: simple concatenation
            self._last_synthesis = self._fallback_synthesis(candidates)
            return [
                AgentDecision(
                    agent_name=self.name,
                    memory_id=mem.id,
                    action="keep",
                    confidence=0.5,
                    reasoning=f"Fallback synthesis used: {str(e)[:50]}",
                )
                for mem in candidates
            ]

    def _build_user_prompt(
        self,
        query: str,
        candidates: list[Memory],
        confidence_map: dict[str, float],
    ) -> str:
        """Build user prompt for synthesis.

        Args:
            query: The user's query.
            candidates: Memories to synthesize.
            confidence_map: Map of memory_id to confidence scores.

        Returns:
            Formatted user prompt.
        """
        # Format candidates with confidence
        formatted = []
        for mem in candidates:
            conf = confidence_map.get(mem.id, 0.7)
            formatted.append(
                f"Memory (ID: {mem.id}, confidence: {conf:.2f}):\n"
                f"  Content: {mem.content}\n"
                f"  Source: {mem.source}\n"
                f"  Timestamp: {mem.timestamp.isoformat()}"
            )

        candidates_text = "\n\n".join(formatted)

        return f"""Query: "{query}"

Memories to synthesize:

{candidates_text}

Settings:
- Max tokens: {self.max_tokens}
- Grouping strategy: {self.grouping_strategy}
- Include attribution: {self.include_attribution}
- Include timestamps: {self.include_timestamps}
- Summarization level: {self.summarization_level} (0=verbose, 1=minimal)

Synthesize these memories into coherent context that answers the query.
Return your synthesis as JSON."""

    def _parse_synthesis_response(
        self,
        llm_response: str,
        candidates: list[Memory],
    ) -> tuple[list[AgentDecision], str]:
        """Parse LLM synthesis response.

        Args:
            llm_response: Raw LLM response.
            candidates: Original candidates.

        Returns:
            Tuple of (decisions, synthesized_context).
        """
        import json

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

            # Get final context
            synthesis = data.get("final_context", "")

            # Parse decisions
            decisions = []
            decision_map = {d["memory_id"]: d for d in data.get("decisions", [])}

            for mem in candidates:
                if mem.id in decision_map:
                    d = decision_map[mem.id]
                    decisions.append(AgentDecision(
                        agent_name=self.name,
                        memory_id=mem.id,
                        action=d.get("action", "merge"),
                        confidence=float(d.get("confidence", 0.8)),
                        reasoning=d.get("reasoning", "Included in synthesis"),
                    ))
                else:
                    decisions.append(AgentDecision(
                        agent_name=self.name,
                        memory_id=mem.id,
                        action="merge",
                        confidence=0.7,
                        reasoning="Included in synthesis",
                    ))

            return decisions, synthesis

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"SynthesisAgent parse error: {e}")
            # Fallback
            synthesis = self._fallback_synthesis(candidates)
            decisions = [
                AgentDecision(
                    agent_name=self.name,
                    memory_id=mem.id,
                    action="keep",
                    confidence=0.5,
                    reasoning=f"Parse error, simple concat: {str(e)[:50]}",
                )
                for mem in candidates
            ]
            return decisions, synthesis

    def _fallback_synthesis(self, candidates: list[Memory]) -> str:
        """Create simple fallback synthesis.

        Args:
            candidates: Memories to synthesize.

        Returns:
            Simple concatenation of memory contents.
        """
        parts = []
        for mem in candidates:
            timestamp = mem.timestamp.strftime("%Y-%m-%d") if self.include_timestamps else ""
            source = f" [Source: {mem.source}]" if self.include_attribution else ""
            if timestamp:
                parts.append(f"({timestamp}) {mem.content}{source}")
            else:
                parts.append(f"{mem.content}{source}")

        synthesis = "\n\n".join(parts)

        # Truncate if over budget
        estimated_tokens = self._estimate_tokens(synthesis)
        if estimated_tokens > self.max_tokens:
            # Rough truncation
            ratio = self.max_tokens / estimated_tokens
            char_limit = int(len(synthesis) * ratio * 0.9)  # 10% buffer
            synthesis = synthesis[:char_limit] + "... [truncated]"

        return synthesis

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count.
        """
        # Rough approximation: ~0.75 words per token
        words = len(text.split())
        return int(words / 0.75)

    def get_last_synthesis(self) -> Optional[str]:
        """Get the most recent synthesized context.

        Returns:
            The last synthesized context string, or None.
        """
        return self._last_synthesis
