"""Chain-of-thought reasoning templates."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ReasoningStep:
    """A single step in a chain-of-thought reasoning process."""

    step_type: str  # "observation", "analysis", "decision", "confidence", "summary"
    content: str
    metadata: Optional[dict] = None


class ChainOfThought:
    """Structured chain-of-thought reasoning pattern."""

    STEP_TYPES = ["observation", "analysis", "decision", "confidence", "summary"]

    def __init__(self):
        """Initialize chain-of-thought container."""
        self.steps: list[ReasoningStep] = []

    def add_observation(self, content: str, metadata: Optional[dict] = None) -> "ChainOfThought":
        """Add an observation step.

        Args:
            content: What was observed.
            metadata: Optional additional data.

        Returns:
            Self for chaining.
        """
        self.steps.append(ReasoningStep("observation", content, metadata))
        return self

    def add_analysis(self, content: str, metadata: Optional[dict] = None) -> "ChainOfThought":
        """Add an analysis step.

        Args:
            content: Analysis of observations.
            metadata: Optional additional data.

        Returns:
            Self for chaining.
        """
        self.steps.append(ReasoningStep("analysis", content, metadata))
        return self

    def add_decision(self, content: str, metadata: Optional[dict] = None) -> "ChainOfThought":
        """Add a decision step.

        Args:
            content: Decision made based on analysis.
            metadata: Optional additional data.

        Returns:
            Self for chaining.
        """
        self.steps.append(ReasoningStep("decision", content, metadata))
        return self

    def add_confidence(
        self, score: float, reasoning: str, metadata: Optional[dict] = None
    ) -> "ChainOfThought":
        """Add a confidence assessment step.

        Args:
            score: Confidence score (0-1).
            reasoning: Why this confidence level.
            metadata: Optional additional data.

        Returns:
            Self for chaining.
        """
        content = f"Confidence: {score:.2f} - {reasoning}"
        self.steps.append(ReasoningStep("confidence", content, metadata or {"score": score}))
        return self

    def add_summary(self, content: str, metadata: Optional[dict] = None) -> "ChainOfThought":
        """Add a summary step.

        Args:
            content: Summary of reasoning.
            metadata: Optional additional data.

        Returns:
            Self for chaining.
        """
        self.steps.append(ReasoningStep("summary", content, metadata))
        return self

    def to_text(self) -> str:
        """Convert reasoning chain to text format.

        Returns:
            Formatted text representation.
        """
        lines = []
        for step in self.steps:
            prefix = step.step_type.upper()
            lines.append(f"[{prefix}] {step.content}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary format.

        Returns:
            Dictionary representation.
        """
        return {
            "steps": [
                {
                    "type": s.step_type,
                    "content": s.content,
                    "metadata": s.metadata,
                }
                for s in self.steps
            ]
        }

    @classmethod
    def from_text(cls, text: str) -> "ChainOfThought":
        """Parse reasoning chain from text.

        Args:
            text: Text representation.

        Returns:
            ChainOfThought instance.
        """
        cot = cls()
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Try to parse [TYPE] content format
            if line.startswith("[") and "]" in line:
                bracket_end = line.index("]")
                step_type = line[1:bracket_end].lower()
                content = line[bracket_end + 1:].strip()

                if step_type in cls.STEP_TYPES:
                    cot.steps.append(ReasoningStep(step_type, content))

        return cot

    def get_final_decision(self) -> Optional[str]:
        """Get the final decision from the chain.

        Returns:
            Decision content or None.
        """
        for step in reversed(self.steps):
            if step.step_type == "decision":
                return step.content
        return None

    def get_confidence(self) -> Optional[float]:
        """Get the confidence score from the chain.

        Returns:
            Confidence score or None.
        """
        for step in reversed(self.steps):
            if step.step_type == "confidence" and step.metadata:
                return step.metadata.get("score")
        return None

    def __len__(self) -> int:
        """Return number of steps."""
        return len(self.steps)

    def __str__(self) -> str:
        """String representation."""
        return self.to_text()


# Template patterns for agent prompts
REASONING_TEMPLATE = """
Follow this reasoning pattern:

1. OBSERVATION: What do you observe about the input?
2. ANALYSIS: What does this mean in context?
3. DECISION: What action should be taken?
4. CONFIDENCE: How confident are you? (0.0-1.0)
5. SUMMARY: Brief summary of your reasoning.

Example:
[OBSERVATION] The memory mentions "Python performance" and the query asks about "Python optimization"
[ANALYSIS] Both refer to improving Python code execution speed, but the memory is about Monty Python comedy, not the programming language
[DECISION] Discard this memory as irrelevant - it's a false positive from embedding similarity
[CONFIDENCE] 0.95 - Very confident this is about the comedy show based on context clues
[SUMMARY] Memory about Monty Python comedy show incorrectly matched Python programming query
"""


def create_reasoning_prompt(
    agent_type: str,
    query: str,
    context: str,
) -> str:
    """Create a reasoning prompt for an agent.

    Args:
        agent_type: Type of agent ("relevance", "recency", "conflict", "synthesis").
        query: The user's query.
        context: Formatted context (memories, etc.).

    Returns:
        Complete prompt with reasoning template.
    """
    agent_guidance = {
        "relevance": "Focus on whether memories are genuinely relevant to the query intent, not just surface similarity.",
        "recency": "Focus on temporal aspects: is the information current, stale, or superseded?",
        "conflict": "Focus on detecting contradictions and determining which information should be preferred.",
        "synthesis": "Focus on combining information coherently while maintaining accuracy.",
    }

    guidance = agent_guidance.get(agent_type, "Apply careful reasoning to the task.")

    return f"""
{REASONING_TEMPLATE}

Agent Focus: {guidance}

Query: {query}

Context:
{context}

Apply the reasoning pattern above to evaluate this input.
"""
