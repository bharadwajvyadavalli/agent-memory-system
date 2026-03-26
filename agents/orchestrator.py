"""Orchestrator: Coordinates the multi-agent reasoning pipeline."""

import asyncio
import logging
import time
from typing import Optional

from agents.base import BaseMemoryAgent
from agents.relevance import RelevanceAgent
from agents.recency import RecencyAgent
from agents.conflict import ConflictAgent
from agents.synthesis import SynthesisAgent
from memory.schema import AgentDecision, Memory, MemoryQuery, RetrievalResult

logger = logging.getLogger(__name__)


class Orchestrator:
    """Orchestrates the multi-agent memory reasoning pipeline."""

    def __init__(
        self,
        relevance_agent: Optional[RelevanceAgent] = None,
        recency_agent: Optional[RecencyAgent] = None,
        conflict_agent: Optional[ConflictAgent] = None,
        synthesis_agent: Optional[SynthesisAgent] = None,
        parallel_first_pass: bool = True,
        agent_order: Optional[list[str]] = None,
        skip_agents: Optional[list[str]] = None,
    ):
        """Initialize the orchestrator.

        Args:
            relevance_agent: RelevanceAgent instance (or None to create default).
            recency_agent: RecencyAgent instance (or None to create default).
            conflict_agent: ConflictAgent instance (or None to create default).
            synthesis_agent: SynthesisAgent instance (or None to create default).
            parallel_first_pass: Run relevance and recency in parallel.
            agent_order: Custom agent execution order.
            skip_agents: Agents to skip (for fast retrieval).
        """
        self.agents: dict[str, BaseMemoryAgent] = {
            "relevance": relevance_agent or RelevanceAgent(),
            "recency": recency_agent or RecencyAgent(),
            "conflict": conflict_agent or ConflictAgent(),
            "synthesis": synthesis_agent or SynthesisAgent(),
        }

        self.parallel_first_pass = parallel_first_pass
        self.agent_order = agent_order or ["relevance", "recency", "conflict", "synthesis"]
        self.skip_agents = set(skip_agents or [])

        self._all_decisions: list[AgentDecision] = []

    async def run(
        self,
        query: MemoryQuery,
        candidates: list[Memory],
    ) -> RetrievalResult:
        """Run the full agent pipeline.

        Args:
            query: The memory query.
            candidates: Initial candidate memories from embedding search.

        Returns:
            RetrievalResult with curated memories and agent decisions.
        """
        start_time = time.time()
        self._all_decisions = []

        if not query.require_reasoning:
            # Fast path: skip agents
            return self._fast_result(candidates, start_time)

        current_candidates = candidates.copy()
        context: dict = {"decisions": []}

        # Execute agent pipeline
        for agent_name in self.agent_order:
            if agent_name in self.skip_agents:
                logger.debug(f"Skipping agent: {agent_name}")
                continue

            agent = self.agents.get(agent_name)
            if not agent:
                logger.warning(f"Unknown agent: {agent_name}")
                continue

            if not current_candidates:
                logger.info(f"No candidates remaining, stopping pipeline at {agent_name}")
                break

            # Run agent
            logger.info(f"Running {agent_name} on {len(current_candidates)} candidates")

            if self.parallel_first_pass and agent_name in ["relevance", "recency"]:
                # Handle parallel execution of first-pass agents
                decisions = await self._run_parallel_first_pass(
                    query.query, current_candidates, context
                )
                # Skip recency since we ran both
                if agent_name == "relevance":
                    self.skip_agents.add("recency")
            else:
                decisions = await agent.reason(query.query, current_candidates, context)

            self._all_decisions.extend(decisions)
            context["decisions"] = self._all_decisions

            # Filter candidates based on decisions
            current_candidates = self._filter_candidates(current_candidates, decisions)

        # Get synthesized context
        synthesis_agent = self.agents.get("synthesis")
        if isinstance(synthesis_agent, SynthesisAgent):
            final_context = synthesis_agent.get_last_synthesis() or ""
        else:
            final_context = self._simple_context(current_candidates)

        # Build result
        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            memories=current_candidates,
            agent_decisions=self._all_decisions,
            final_context=final_context,
            metadata={
                "latency_ms": latency_ms,
                "candidates_considered": len(candidates),
                "candidates_filtered": len(candidates) - len(current_candidates),
                "agents_used": [a for a in self.agent_order if a not in self.skip_agents],
            },
        )

    async def run_parallel(
        self,
        query: MemoryQuery,
        candidates: list[Memory],
    ) -> RetrievalResult:
        """Run pipeline with parallel first-pass agents.

        This is an alias for run() with parallel_first_pass=True.

        Args:
            query: The memory query.
            candidates: Initial candidates.

        Returns:
            RetrievalResult.
        """
        self.parallel_first_pass = True
        return await self.run(query, candidates)

    def run_sync(
        self,
        query: MemoryQuery,
        candidates: list[Memory],
    ) -> RetrievalResult:
        """Synchronous wrapper for run().

        Args:
            query: The memory query.
            candidates: Initial candidates.

        Returns:
            RetrievalResult.
        """
        return asyncio.run(self.run(query, candidates))

    async def _run_parallel_first_pass(
        self,
        query: str,
        candidates: list[Memory],
        context: dict,
    ) -> list[AgentDecision]:
        """Run relevance and recency agents in parallel.

        Args:
            query: The query string.
            candidates: Current candidates.
            context: Context for agents.

        Returns:
            Merged decisions from both agents.
        """
        relevance_agent = self.agents.get("relevance")
        recency_agent = self.agents.get("recency")

        if not relevance_agent or not recency_agent:
            return []

        # Run in parallel
        relevance_task = relevance_agent.reason(query, candidates, context)
        recency_task = recency_agent.reason(query, candidates, context)

        relevance_decisions, recency_decisions = await asyncio.gather(
            relevance_task, recency_task
        )

        # Merge decisions: both must keep for memory to survive
        merged = []
        relevance_map = {d.memory_id: d for d in relevance_decisions}
        recency_map = {d.memory_id: d for d in recency_decisions}

        for mem in candidates:
            rel_dec = relevance_map.get(mem.id)
            rec_dec = recency_map.get(mem.id)

            if not rel_dec or not rec_dec:
                continue

            # Combine decisions
            if rel_dec.action == "discard" or rec_dec.action == "discard":
                action = "discard"
            elif rel_dec.action == "flag_conflict" or rec_dec.action == "flag_conflict":
                action = "flag_conflict"
            else:
                action = "keep"

            # Average confidence
            confidence = (rel_dec.confidence + rec_dec.confidence) / 2

            # Combine reasoning
            reasoning = f"Relevance: {rel_dec.reasoning} | Recency: {rec_dec.reasoning}"

            merged.append(AgentDecision(
                agent_name="relevance+recency",
                memory_id=mem.id,
                action=action,
                confidence=confidence,
                reasoning=reasoning,
            ))

        return merged

    def _filter_candidates(
        self,
        candidates: list[Memory],
        decisions: list[AgentDecision],
    ) -> list[Memory]:
        """Filter candidates based on agent decisions.

        Args:
            candidates: Current candidate memories.
            decisions: Decisions from an agent.

        Returns:
            Filtered list of candidates (those with keep or merge action).
        """
        decision_map = {d.memory_id: d for d in decisions}

        filtered = []
        for mem in candidates:
            decision = decision_map.get(mem.id)
            if decision and decision.action in ["keep", "merge", "flag_conflict"]:
                filtered.append(mem)

        return filtered

    def _fast_result(
        self,
        candidates: list[Memory],
        start_time: float,
    ) -> RetrievalResult:
        """Create fast result without agent reasoning.

        Args:
            candidates: Candidates to return.
            start_time: Pipeline start time.

        Returns:
            Simple RetrievalResult.
        """
        return RetrievalResult(
            memories=candidates,
            agent_decisions=[],
            final_context=self._simple_context(candidates),
            metadata={
                "latency_ms": (time.time() - start_time) * 1000,
                "candidates_considered": len(candidates),
                "candidates_filtered": 0,
                "agents_used": [],
                "mode": "fast",
            },
        )

    def _simple_context(self, memories: list[Memory]) -> str:
        """Create simple concatenated context.

        Args:
            memories: Memories to include.

        Returns:
            Simple text context.
        """
        parts = [f"[{m.source}] {m.content}" for m in memories]
        return "\n\n".join(parts)

    def get_all_decisions(self) -> list[AgentDecision]:
        """Get all decisions from the last run.

        Returns:
            List of all AgentDecision objects.
        """
        return self._all_decisions.copy()

    def configure(
        self,
        agent_order: Optional[list[str]] = None,
        skip_agents: Optional[list[str]] = None,
        parallel_first_pass: Optional[bool] = None,
    ) -> None:
        """Update orchestrator configuration.

        Args:
            agent_order: New agent execution order.
            skip_agents: Agents to skip.
            parallel_first_pass: Whether to run first-pass in parallel.
        """
        if agent_order is not None:
            self.agent_order = agent_order
        if skip_agents is not None:
            self.skip_agents = set(skip_agents)
        if parallel_first_pass is not None:
            self.parallel_first_pass = parallel_first_pass
