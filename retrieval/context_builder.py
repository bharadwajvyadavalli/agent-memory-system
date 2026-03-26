"""Context builder for assembling final context from memories."""

import logging
from typing import Optional

from memory.schema import AgentDecision, Memory

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds final context string from curated memories."""

    def __init__(
        self,
        max_tokens: int = 4096,
        include_timestamps: bool = True,
        include_source: bool = True,
        include_confidence: bool = False,
        separator: str = "\n\n---\n\n",
        truncation_strategy: str = "confidence",
    ):
        """Initialize the context builder.

        Args:
            max_tokens: Maximum tokens in output.
            include_timestamps: Include memory timestamps.
            include_source: Include source attribution.
            include_confidence: Include confidence scores.
            separator: Separator between memories.
            truncation_strategy: How to truncate ("confidence", "recency", "proportional").
        """
        self.max_tokens = max_tokens
        self.include_timestamps = include_timestamps
        self.include_source = include_source
        self.include_confidence = include_confidence
        self.separator = separator
        self.truncation_strategy = truncation_strategy

    def build(
        self,
        memories: list[Memory],
        decisions: Optional[list[AgentDecision]] = None,
        max_tokens: Optional[int] = None,
        synthesis: Optional[str] = None,
    ) -> str:
        """Build context string from memories.

        Args:
            memories: List of memories to include.
            decisions: Optional agent decisions for confidence scores.
            max_tokens: Override max tokens.
            synthesis: Pre-built synthesis from SynthesisAgent.

        Returns:
            Formatted context string.
        """
        if synthesis:
            # Use pre-built synthesis if available
            return self._truncate_to_budget(synthesis, max_tokens or self.max_tokens)

        if not memories:
            return ""

        # Build confidence map
        confidence_map = {}
        if decisions:
            for d in decisions:
                if d.memory_id not in confidence_map or d.confidence > confidence_map[d.memory_id]:
                    confidence_map[d.memory_id] = d.confidence

        # Format each memory
        formatted_parts = []
        for mem in memories:
            formatted = self._format_memory(mem, confidence_map.get(mem.id))
            formatted_parts.append((mem, formatted))

        # Apply token budget
        budget = max_tokens or self.max_tokens
        final_parts = self._apply_budget(formatted_parts, budget, confidence_map)

        return self.separator.join(final_parts)

    def _format_memory(
        self,
        memory: Memory,
        confidence: Optional[float] = None,
    ) -> str:
        """Format a single memory for context.

        Args:
            memory: Memory to format.
            confidence: Optional confidence score.

        Returns:
            Formatted string.
        """
        parts = []

        # Header with metadata
        header_parts = []
        if self.include_source:
            header_parts.append(f"[Source: {memory.source}]")
        if self.include_timestamps:
            header_parts.append(f"[{memory.timestamp.strftime('%Y-%m-%d')}]")
        if self.include_confidence and confidence is not None:
            header_parts.append(f"[Confidence: {confidence:.2f}]")

        if header_parts:
            parts.append(" ".join(header_parts))

        # Content
        parts.append(memory.content)

        return "\n".join(parts)

    def _apply_budget(
        self,
        formatted_parts: list[tuple[Memory, str]],
        budget: int,
        confidence_map: dict[str, float],
    ) -> list[str]:
        """Apply token budget, truncating as needed.

        Args:
            formatted_parts: List of (Memory, formatted_string) tuples.
            budget: Token budget.
            confidence_map: Confidence scores for prioritization.

        Returns:
            List of formatted strings within budget.
        """
        # Estimate current tokens
        all_text = self.separator.join(p[1] for p in formatted_parts)
        current_tokens = self._estimate_tokens(all_text)

        if current_tokens <= budget:
            return [p[1] for p in formatted_parts]

        # Need to truncate
        if self.truncation_strategy == "confidence":
            # Sort by confidence, keep highest
            sorted_parts = sorted(
                formatted_parts,
                key=lambda x: confidence_map.get(x[0].id, 0.5),
                reverse=True,
            )
        elif self.truncation_strategy == "recency":
            # Sort by timestamp, keep newest
            sorted_parts = sorted(
                formatted_parts,
                key=lambda x: x[0].timestamp,
                reverse=True,
            )
        else:
            # Proportional - keep original order
            sorted_parts = formatted_parts

        # Add parts until budget exhausted
        result = []
        used_tokens = 0
        separator_tokens = self._estimate_tokens(self.separator)

        for _, formatted in sorted_parts:
            part_tokens = self._estimate_tokens(formatted)
            if used_tokens + part_tokens + separator_tokens > budget:
                # Try to add truncated version
                remaining = budget - used_tokens - separator_tokens
                if remaining > 50:  # Worth adding truncated
                    truncated = self._truncate_text(formatted, remaining)
                    result.append(truncated)
                break
            result.append(formatted)
            used_tokens += part_tokens + separator_tokens

        return result

    def _truncate_to_budget(self, text: str, budget: int) -> str:
        """Truncate text to fit budget.

        Args:
            text: Text to truncate.
            budget: Token budget.

        Returns:
            Truncated text.
        """
        current = self._estimate_tokens(text)
        if current <= budget:
            return text

        return self._truncate_text(text, budget)

    def _truncate_text(self, text: str, target_tokens: int) -> str:
        """Truncate text to target tokens.

        Args:
            text: Text to truncate.
            target_tokens: Target token count.

        Returns:
            Truncated text with ellipsis.
        """
        # Rough: 0.75 words per token
        target_words = int(target_tokens * 0.75)
        words = text.split()

        if len(words) <= target_words:
            return text

        truncated = " ".join(words[:target_words])
        return truncated + "..."

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count.

        Args:
            text: Text to estimate.

        Returns:
            Estimated tokens.
        """
        # Rough approximation
        return int(len(text.split()) / 0.75)

    def build_minimal(self, memories: list[Memory]) -> str:
        """Build minimal context (content only).

        Args:
            memories: Memories to include.

        Returns:
            Minimal context string.
        """
        return "\n\n".join(m.content for m in memories)

    def build_structured(
        self,
        memories: list[Memory],
        decisions: Optional[list[AgentDecision]] = None,
    ) -> dict:
        """Build structured context as dict.

        Args:
            memories: Memories to include.
            decisions: Optional agent decisions.

        Returns:
            Dict with structured context.
        """
        confidence_map = {}
        if decisions:
            for d in decisions:
                confidence_map[d.memory_id] = d.confidence

        return {
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "source": m.source,
                    "timestamp": m.timestamp.isoformat(),
                    "confidence": confidence_map.get(m.id, 0.5),
                }
                for m in memories
            ],
            "count": len(memories),
            "text": self.build(memories, decisions),
        }
