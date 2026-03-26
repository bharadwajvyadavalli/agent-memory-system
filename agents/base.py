"""Base class for memory reasoning agents."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from memory.schema import AgentDecision, Memory

logger = logging.getLogger(__name__)


class BaseMemoryAgent(ABC):
    """Abstract base class for all memory reasoning agents."""

    def __init__(
        self,
        llm_provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the agent.

        Args:
            llm_provider: LLM provider ("openai", "anthropic", "mock").
            model_name: Model name to use.
            temperature: Sampling temperature.
            max_retries: Maximum retry attempts for LLM calls.
            retry_delay: Base delay between retries (exponential backoff).
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: Optional[Any] = None

        if llm_provider != "mock":
            self._init_client()

    def _init_client(self) -> None:
        """Initialize the LLM client."""
        if self.llm_provider == "openai":
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI()
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

        elif self.llm_provider == "anthropic":
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic()
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")

    @property
    def name(self) -> str:
        """Return the agent's name."""
        return self.__class__.__name__

    @abstractmethod
    async def reason(
        self,
        query: str,
        candidates: list[Memory],
        context: Optional[dict] = None,
    ) -> list[AgentDecision]:
        """Perform reasoning over candidate memories.

        Args:
            query: The user's query.
            candidates: List of candidate memories to evaluate.
            context: Optional additional context (e.g., from previous agents).

        Returns:
            List of AgentDecision objects, one per candidate.
        """
        pass

    def reason_sync(
        self,
        query: str,
        candidates: list[Memory],
        context: Optional[dict] = None,
    ) -> list[AgentDecision]:
        """Synchronous wrapper for reason().

        Args:
            query: The user's query.
            candidates: List of candidate memories.
            context: Optional additional context.

        Returns:
            List of AgentDecision objects.
        """
        return asyncio.run(self.reason(query, candidates, context))

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Make an LLM call with retry logic.

        Args:
            system_prompt: System message.
            user_prompt: User message.

        Returns:
            The LLM's response text.

        Raises:
            Exception: If all retries fail.
        """
        if self.llm_provider == "mock":
            return self._mock_response(user_prompt)

        last_error = None

        for attempt in range(self.max_retries):
            try:
                if self.llm_provider == "openai":
                    response = await self._client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                    )
                    return response.choices[0].message.content

                elif self.llm_provider == "anthropic":
                    response = await self._client.messages.create(
                        model=self.model_name,
                        max_tokens=1024,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                    return response.content[0].text

            except Exception as e:
                last_error = e
                logger.warning(
                    f"{self.name} LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        raise last_error or Exception("LLM call failed")

    def _mock_response(self, user_prompt: str) -> str:
        """Generate a mock response for testing.

        Args:
            user_prompt: The user prompt.

        Returns:
            A mock JSON response.
        """
        # Default mock: keep all with high confidence
        return json.dumps({
            "decisions": [
                {
                    "memory_id": "mock_id",
                    "action": "keep",
                    "confidence": 0.8,
                    "reasoning": "Mock reasoning for testing",
                }
            ]
        })

    def _parse_decisions(
        self,
        llm_response: str,
        candidates: list[Memory],
    ) -> list[AgentDecision]:
        """Parse LLM JSON response into AgentDecision objects.

        Args:
            llm_response: Raw LLM response text.
            candidates: The original candidate memories.

        Returns:
            List of AgentDecision objects.
        """
        try:
            # Try to extract JSON from response
            response_text = llm_response.strip()

            # Handle markdown code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            data = json.loads(response_text)

            decisions = []
            decision_list = data.get("decisions", [])

            # Create a map of memory_id to decision
            decision_map = {d.get("memory_id"): d for d in decision_list}

            for candidate in candidates:
                if candidate.id in decision_map:
                    d = decision_map[candidate.id]
                    decisions.append(AgentDecision(
                        agent_name=self.name,
                        memory_id=candidate.id,
                        action=d.get("action", "keep"),
                        confidence=float(d.get("confidence", 0.5)),
                        reasoning=d.get("reasoning", "No reasoning provided"),
                    ))
                else:
                    # Candidate not in response - default to keep with low confidence
                    decisions.append(AgentDecision(
                        agent_name=self.name,
                        memory_id=candidate.id,
                        action="keep",
                        confidence=0.3,
                        reasoning="Memory not evaluated by agent - keeping by default",
                    ))

            return decisions

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"{self.name} failed to parse LLM response: {e}")
            # Graceful fallback: keep all with low confidence
            return [
                AgentDecision(
                    agent_name=self.name,
                    memory_id=candidate.id,
                    action="keep",
                    confidence=0.3,
                    reasoning=f"Parse error, keeping by default: {str(e)[:100]}",
                )
                for candidate in candidates
            ]

    def _format_candidates_for_prompt(self, candidates: list[Memory]) -> str:
        """Format candidate memories for inclusion in a prompt.

        Args:
            candidates: List of memories.

        Returns:
            Formatted string representation.
        """
        formatted = []
        for i, mem in enumerate(candidates, 1):
            formatted.append(
                f"Memory {i} (ID: {mem.id}):\n"
                f"  Content: {mem.content}\n"
                f"  Source: {mem.source}\n"
                f"  Timestamp: {mem.timestamp.isoformat()}\n"
                f"  Tags: {', '.join(mem.tags) if mem.tags else 'none'}"
            )
        return "\n\n".join(formatted)
