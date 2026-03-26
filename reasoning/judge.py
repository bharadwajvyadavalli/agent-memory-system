"""MetaJudge: Evaluates agent decision quality."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from memory.schema import AgentDecision, Memory

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result of a judge evaluation."""

    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    quality_score: Optional[float] = None  # 1-5 scale for LLM evaluation
    reasoning: str = ""
    should_escalate: bool = False
    metadata: Optional[dict] = None


class MetaJudge:
    """Meta-judge that evaluates agent decision quality."""

    JUDGE_SYSTEM_PROMPT = """You are a meta-judge evaluating the quality of memory retrieval decisions.

Given:
- A query
- Candidate memories
- Decisions made by retrieval agents
- (Optionally) Ground truth correct answers

Evaluate the decisions on a 1-5 scale:
1 - Very poor: Critical relevant memories missed or irrelevant ones kept
2 - Poor: Several mistakes in relevance/recency/conflict handling
3 - Acceptable: Some minor errors but generally correct
4 - Good: Mostly correct with minimal issues
5 - Excellent: Perfect or near-perfect decisions

Output JSON:
{
  "quality_score": 1-5,
  "reasoning": "Detailed explanation",
  "issues": ["list of specific issues if any"],
  "should_escalate": true/false (recommend human review if quality < 3)
}"""

    def __init__(
        self,
        llm_provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        escalation_threshold: float = 0.5,
    ):
        """Initialize the meta-judge.

        Args:
            llm_provider: LLM provider for quality evaluation.
            model_name: Model name.
            escalation_threshold: Average confidence below this triggers escalation.
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.escalation_threshold = escalation_threshold
        self._client = None

        if llm_provider != "mock":
            self._init_client()

    def _init_client(self) -> None:
        """Initialize the LLM client."""
        if self.llm_provider == "openai":
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI()
            except ImportError:
                raise ImportError("openai package required")
        elif self.llm_provider == "anthropic":
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic()
            except ImportError:
                raise ImportError("anthropic package required")

    def evaluate_decisions(
        self,
        query: str,
        candidates: list[Memory],
        decisions: list[AgentDecision],
        ground_truth: Optional[list[str]] = None,
    ) -> JudgeResult:
        """Evaluate agent decisions.

        Args:
            query: The original query.
            candidates: Candidate memories.
            decisions: Decisions made by agents.
            ground_truth: Optional list of correct memory IDs.

        Returns:
            JudgeResult with evaluation metrics.
        """
        return asyncio.run(
            self.aevaluate_decisions(query, candidates, decisions, ground_truth)
        )

    async def aevaluate_decisions(
        self,
        query: str,
        candidates: list[Memory],
        decisions: list[AgentDecision],
        ground_truth: Optional[list[str]] = None,
    ) -> JudgeResult:
        """Async version of evaluate_decisions.

        Args:
            query: The original query.
            candidates: Candidate memories.
            decisions: Decisions made by agents.
            ground_truth: Optional list of correct memory IDs.

        Returns:
            JudgeResult with evaluation metrics.
        """
        result = JudgeResult(metadata={})

        # Compute metrics if ground truth available
        if ground_truth:
            kept_ids = {d.memory_id for d in decisions if d.action in ["keep", "merge"]}
            truth_set = set(ground_truth)

            true_positives = len(kept_ids & truth_set)
            false_positives = len(kept_ids - truth_set)
            false_negatives = len(truth_set - kept_ids)

            result.precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0.0
            )
            result.recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0.0
            )
            result.f1 = (
                2 * result.precision * result.recall / (result.precision + result.recall)
                if (result.precision + result.recall) > 0
                else 0.0
            )

        # Check for escalation based on confidence
        result.should_escalate = self.should_escalate(decisions)

        # Get LLM quality evaluation
        if self.llm_provider != "mock":
            try:
                llm_result = await self._llm_evaluate(query, candidates, decisions, ground_truth)
                result.quality_score = llm_result.get("quality_score")
                result.reasoning = llm_result.get("reasoning", "")
                if llm_result.get("should_escalate"):
                    result.should_escalate = True
            except Exception as e:
                logger.warning(f"LLM evaluation failed: {e}")
                result.reasoning = f"LLM evaluation failed: {e}"
        else:
            # Mock evaluation
            result.quality_score = 4.0
            result.reasoning = "Mock evaluation: decisions appear reasonable"

        return result

    def should_escalate(self, decisions: list[AgentDecision]) -> bool:
        """Determine if decisions should be escalated for review.

        Args:
            decisions: List of agent decisions.

        Returns:
            True if escalation recommended.
        """
        if not decisions:
            return True

        avg_confidence = sum(d.confidence for d in decisions) / len(decisions)
        return avg_confidence < self.escalation_threshold

    async def _llm_evaluate(
        self,
        query: str,
        candidates: list[Memory],
        decisions: list[AgentDecision],
        ground_truth: Optional[list[str]],
    ) -> dict:
        """Get LLM quality evaluation.

        Args:
            query: The original query.
            candidates: Candidate memories.
            decisions: Agent decisions.
            ground_truth: Optional ground truth.

        Returns:
            Dict with quality evaluation.
        """
        # Format prompt
        candidates_text = "\n".join(
            f"- {m.id}: {m.content[:100]}..." for m in candidates
        )
        decisions_text = "\n".join(
            f"- {d.memory_id}: {d.action} (confidence: {d.confidence:.2f}) - {d.reasoning[:50]}..."
            for d in decisions
        )

        user_prompt = f"""Query: "{query}"

Candidates:
{candidates_text}

Decisions:
{decisions_text}

Ground Truth IDs: {ground_truth if ground_truth else "Not provided"}

Evaluate the quality of these decisions."""

        try:
            import json

            if self.llm_provider == "openai":
                response = await self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                )
                content = response.choices[0].message.content
            elif self.llm_provider == "anthropic":
                response = await self._client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    system=self.JUDGE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                content = response.content[0].text
            else:
                return {"quality_score": 4, "reasoning": "Mock"}

            # Parse JSON response
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            return json.loads(content)

        except Exception as e:
            logger.warning(f"LLM evaluation parse error: {e}")
            return {"quality_score": 3, "reasoning": f"Parse error: {e}"}

    def evaluate_conflict_resolution(
        self,
        conflicts: list,
        ground_truth_winners: dict[tuple[str, str], str],
    ) -> float:
        """Evaluate conflict resolution accuracy.

        Args:
            conflicts: List of ConflictRecord objects.
            ground_truth_winners: Dict mapping (id_a, id_b) to correct winner_id.

        Returns:
            Accuracy score (0-1).
        """
        if not conflicts or not ground_truth_winners:
            return 0.0

        correct = 0
        total = 0

        for conflict in conflicts:
            key = (conflict.memory_a_id, conflict.memory_b_id)
            alt_key = (conflict.memory_b_id, conflict.memory_a_id)

            truth = ground_truth_winners.get(key) or ground_truth_winners.get(alt_key)
            if truth:
                total += 1
                if conflict.winner_id == truth:
                    correct += 1

        return correct / total if total > 0 else 0.0
