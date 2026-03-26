"""Prompt templates for each agent role."""

from typing import Optional

from memory.schema import Memory


class PromptManager:
    """Manages prompt templates for all agents."""

    # System prompts for each agent
    SYSTEM_PROMPTS = {
        "relevance": """You are a relevance evaluation agent in a memory retrieval system. Your job is to determine if candidate memories are actually relevant to the user's query, going beyond simple keyword or embedding similarity.

Key responsibilities:
- Distinguish genuine relevance from superficial word matches
- Consider the user's intent, not just literal query words
- Identify false positives from embedding similarity

Output JSON format:
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

Example 1:
Query: "Python performance optimization"
Memory: "Monty Python's Flying Circus was a British comedy show from 1969-1974."
Decision: {"memory_id": "...", "action": "discard", "confidence": 0.95, "reasoning": "About comedy show, not programming language"}

Example 2:
Query: "Apple stock price"
Memory: "Apple released the M3 chip with significant performance improvements."
Decision: {"memory_id": "...", "action": "keep", "confidence": 0.7, "reasoning": "Product news affects stock, moderately relevant"}

Example 3:
Query: "best pizza toppings"
Memory: "The Tower of Pisa is a famous Italian landmark."
Decision: {"memory_id": "...", "action": "discard", "confidence": 0.9, "reasoning": "About Italian landmark, not food"}""",

        "recency": """You are a temporal reasoning agent in a memory retrieval system. Your job is to evaluate memories for recency and staleness.

Key responsibilities:
- Assess if information is current or potentially outdated
- Detect temporal language ("as of 2023", "current policy", "was")
- Identify when newer information supersedes older

Actions:
- keep: Memory is current and valid
- discard: Memory is stale or superseded
- flag_conflict: Newer version exists but partial overlap

Output JSON format:
{
  "decisions": [
    {
      "memory_id": "<id>",
      "action": "keep" | "discard" | "flag_conflict",
      "confidence": 0.0-1.0,
      "reasoning": "Explanation including temporal assessment",
      "temporal_signal": "current" | "outdated" | "versioned" | "none"
    }
  ]
}

Example:
Memory A (2022): "Return policy is 30 days"
Memory B (2024): "Updated: Return policy changed to 15 days"
For query "What's the return policy?":
- Memory A: discard (superseded)
- Memory B: keep (most recent, explicitly dated)""",

        "conflict": """You are a conflict detection agent in a memory retrieval system. Your job is to identify contradictions between memories and resolve them.

Conflict types:
- full_contradiction: Direct contradiction, newer wins
- partial_update: One updates part of another, both may be valid
- scope_difference: Different scopes, not actually conflicting

Output JSON format:
{
  "conflicts": [
    {
      "memory_a_id": "<id>",
      "memory_b_id": "<id>",
      "conflict_type": "full_contradiction" | "partial_update" | "scope_difference" | "none",
      "winner_id": "<id>" | null,
      "reasoning": "Explanation"
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

Example:
Memory A: "CEO is John Smith"
Memory B (newer): "Jane Doe appointed CEO"
Conflict: full_contradiction, winner: Memory B
Resolution: A=discard, B=keep""",

        "synthesis": """You are a synthesis agent in a memory retrieval system. Your job is to merge related memories into coherent context.

Key responsibilities:
- Group related memories by topic or entity
- Synthesize each group into natural language
- Include source attribution
- Respect token budget

Output JSON format:
{
  "groups": [
    {
      "topic": "Topic name",
      "memory_ids": ["id1", "id2"],
      "synthesis": "Coherent paragraph with [Source: name] attribution"
    }
  ],
  "final_context": "Complete synthesized context",
  "decisions": [
    {
      "memory_id": "<id>",
      "action": "keep" | "merge",
      "confidence": 0.0-1.0,
      "reasoning": "How memory was used"
    }
  ]
}""",
    }

    def __init__(self):
        """Initialize the prompt manager."""
        self._custom_prompts: dict[str, str] = {}

    def get_system_prompt(self, agent_name: str) -> str:
        """Get system prompt for an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            System prompt string.
        """
        if agent_name in self._custom_prompts:
            return self._custom_prompts[agent_name]
        return self.SYSTEM_PROMPTS.get(agent_name, "You are a helpful assistant.")

    def set_custom_prompt(self, agent_name: str, prompt: str) -> None:
        """Set a custom system prompt for an agent.

        Args:
            agent_name: Name of the agent.
            prompt: Custom system prompt.
        """
        self._custom_prompts[agent_name] = prompt

    def build_user_prompt(
        self,
        agent_name: str,
        query: str,
        candidates: list[Memory],
        context: Optional[dict] = None,
    ) -> str:
        """Build user prompt for an agent.

        Args:
            agent_name: Name of the agent.
            query: The user's query.
            candidates: Candidate memories.
            context: Optional additional context.

        Returns:
            Formatted user prompt.
        """
        candidates_text = self._format_candidates(candidates)

        base_prompt = f"""Query: "{query}"

Candidate memories to evaluate:

{candidates_text}

Evaluate each memory according to your role and return your decisions as JSON."""

        return base_prompt

    def _format_candidates(self, candidates: list[Memory]) -> str:
        """Format candidates for prompt.

        Args:
            candidates: List of memories.

        Returns:
            Formatted string.
        """
        parts = []
        for i, mem in enumerate(candidates, 1):
            parts.append(
                f"Memory {i} (ID: {mem.id}):\n"
                f"  Content: {mem.content}\n"
                f"  Source: {mem.source}\n"
                f"  Timestamp: {mem.timestamp.isoformat()}\n"
                f"  Tags: {', '.join(mem.tags) if mem.tags else 'none'}"
            )
        return "\n\n".join(parts)


# Module-level convenience function
_prompt_manager = PromptManager()


def get_prompt(
    agent_name: str,
    query: str,
    candidates: list[Memory],
    context: Optional[dict] = None,
) -> tuple[str, str]:
    """Get system and user prompts for an agent.

    Args:
        agent_name: Name of the agent.
        query: The user's query.
        candidates: Candidate memories.
        context: Optional additional context.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system = _prompt_manager.get_system_prompt(agent_name)
    user = _prompt_manager.build_user_prompt(agent_name, query, candidates, context)
    return system, user
