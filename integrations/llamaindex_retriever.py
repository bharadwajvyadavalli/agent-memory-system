"""LlamaIndex integration for ASMR retrieval."""

from typing import Any, Optional

from memory.schema import Memory
from retrieval.pipeline import RetrievalPipeline


class ASMRLlamaIndexRetriever:
    """Drop-in LlamaIndex BaseRetriever implementation for ASMR.

    Usage:
        from integrations.llamaindex_retriever import ASMRLlamaIndexRetriever

        retriever = ASMRLlamaIndexRetriever()
        retriever.add_memory("Some content", source="doc1")

        # Use with LlamaIndex
        from llama_index.core.query_engine import RetrieverQueryEngine
        query_engine = RetrieverQueryEngine.from_args(retriever=retriever)
    """

    def __init__(
        self,
        pipeline: Optional[RetrievalPipeline] = None,
        top_k: int = 5,
        require_reasoning: bool = True,
        **kwargs: Any,
    ):
        """Initialize the ASMR LlamaIndex retriever.

        Args:
            pipeline: ASMR RetrievalPipeline instance.
            top_k: Number of nodes to retrieve.
            require_reasoning: Whether to use agent reasoning.
            **kwargs: Additional arguments passed to pipeline.
        """
        self._pipeline = pipeline or RetrievalPipeline(**kwargs)
        self._top_k = top_k
        self._require_reasoning = require_reasoning
        self._callback_manager = None

    def _retrieve(self, query_str: str) -> list:
        """Core retrieval method.

        Args:
            query_str: The search query.

        Returns:
            List of NodeWithScore objects.
        """
        try:
            from llama_index.core.schema import NodeWithScore, TextNode
        except ImportError:
            raise ImportError(
                "llama-index-core is required. Install with: pip install llama-index-core"
            )

        # Retrieve from ASMR
        result = self._pipeline.retrieve(
            query=query_str,
            top_k=self._top_k,
            require_reasoning=self._require_reasoning,
        )

        # Build confidence map from agent decisions
        confidence_map = {}
        for decision in result.agent_decisions:
            if decision.memory_id not in confidence_map:
                confidence_map[decision.memory_id] = decision.confidence

        # Convert to LlamaIndex NodeWithScore
        nodes = []
        for memory in result.memories:
            node = TextNode(
                text=memory.content,
                id_=memory.id,
                metadata={
                    "source": memory.source,
                    "timestamp": memory.timestamp.isoformat(),
                    "tags": memory.tags,
                    **memory.metadata,
                },
            )

            # Use agent confidence as score
            score = confidence_map.get(memory.id, 0.5)

            nodes.append(NodeWithScore(node=node, score=score))

        return nodes

    async def _aretrieve(self, query_str: str) -> list:
        """Async retrieval method.

        Args:
            query_str: The search query.

        Returns:
            List of NodeWithScore objects.
        """
        try:
            from llama_index.core.schema import NodeWithScore, TextNode
        except ImportError:
            raise ImportError(
                "llama-index-core is required. Install with: pip install llama-index-core"
            )

        # Async retrieve from ASMR
        result = await self._pipeline.aretrieve(
            query=query_str,
            top_k=self._top_k,
            require_reasoning=self._require_reasoning,
        )

        # Build confidence map
        confidence_map = {}
        for decision in result.agent_decisions:
            if decision.memory_id not in confidence_map:
                confidence_map[decision.memory_id] = decision.confidence

        # Convert to LlamaIndex NodeWithScore
        nodes = []
        for memory in result.memories:
            node = TextNode(
                text=memory.content,
                id_=memory.id,
                metadata={
                    "source": memory.source,
                    "timestamp": memory.timestamp.isoformat(),
                    "tags": memory.tags,
                    **memory.metadata,
                },
            )

            score = confidence_map.get(memory.id, 0.5)
            nodes.append(NodeWithScore(node=node, score=score))

        return nodes

    # LlamaIndex interface compatibility
    def retrieve(self, query_str: str) -> list:
        """LlamaIndex retriever interface method."""
        return self._retrieve(query_str)

    async def aretrieve(self, query_str: str) -> list:
        """LlamaIndex async retriever interface method."""
        return await self._aretrieve(query_str)

    # Convenience methods
    def add_memory(
        self,
        content: str,
        source: str,
        metadata: Optional[dict] = None,
        tags: Optional[list[str]] = None,
    ) -> Memory:
        """Add a memory to the underlying store.

        Args:
            content: Memory content.
            source: Source identifier.
            metadata: Additional metadata.
            tags: Tags for categorization.

        Returns:
            Created Memory object.
        """
        return self._pipeline.add_memory(
            content=content,
            source=source,
            metadata=metadata,
            tags=tags,
        )

    def add_nodes(self, nodes: list) -> list[str]:
        """Add LlamaIndex nodes to the memory store.

        Args:
            nodes: List of LlamaIndex Node objects.

        Returns:
            List of created memory IDs.
        """
        ids = []
        for node in nodes:
            metadata = node.metadata.copy() if hasattr(node, "metadata") else {}
            source = metadata.pop("source", "llamaindex")
            tags = metadata.pop("tags", [])

            # Get text content
            text = node.get_content() if hasattr(node, "get_content") else str(node)

            memory = self._pipeline.add_memory(
                content=text,
                source=source,
                metadata=metadata,
                tags=tags if isinstance(tags, list) else [],
            )
            ids.append(memory.id)

        return ids

    @property
    def pipeline(self) -> RetrievalPipeline:
        """Get the underlying ASMR pipeline."""
        return self._pipeline

    @property
    def callback_manager(self):
        """Get callback manager (LlamaIndex compatibility)."""
        return self._callback_manager

    @callback_manager.setter
    def callback_manager(self, value):
        """Set callback manager (LlamaIndex compatibility)."""
        self._callback_manager = value


def create_asmr_retriever(
    top_k: int = 5,
    require_reasoning: bool = True,
    **kwargs: Any,
) -> ASMRLlamaIndexRetriever:
    """Factory function to create an ASMR LlamaIndex retriever.

    Args:
        top_k: Number of nodes to retrieve.
        require_reasoning: Whether to use agent reasoning.
        **kwargs: Additional arguments for RetrievalPipeline.

    Returns:
        Configured ASMRLlamaIndexRetriever.
    """
    return ASMRLlamaIndexRetriever(
        top_k=top_k,
        require_reasoning=require_reasoning,
        **kwargs,
    )
