"""LangChain integration for ASMR retrieval."""

from typing import Any, Optional

from memory.schema import Memory
from retrieval.pipeline import RetrievalPipeline


class ASMRLangChainRetriever:
    """Drop-in LangChain BaseRetriever implementation for ASMR.

    Usage:
        from integrations.langchain_retriever import ASMRLangChainRetriever

        retriever = ASMRLangChainRetriever()
        retriever.add_memory("Some content", source="doc1")

        # Use with LangChain
        from langchain.chains import RetrievalQA
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    """

    def __init__(
        self,
        pipeline: Optional[RetrievalPipeline] = None,
        top_k: int = 5,
        require_reasoning: bool = True,
        **kwargs: Any,
    ):
        """Initialize the ASMR LangChain retriever.

        Args:
            pipeline: ASMR RetrievalPipeline instance.
            top_k: Number of documents to retrieve.
            require_reasoning: Whether to use agent reasoning.
            **kwargs: Additional arguments passed to pipeline.
        """
        self._pipeline = pipeline or RetrievalPipeline(**kwargs)
        self._top_k = top_k
        self._require_reasoning = require_reasoning

    def _get_relevant_documents(self, query: str) -> list:
        """Get relevant documents for a query.

        This is the core LangChain retriever interface method.

        Args:
            query: The search query.

        Returns:
            List of LangChain Document objects.
        """
        try:
            from langchain_core.documents import Document
        except ImportError:
            raise ImportError(
                "langchain-core is required. Install with: pip install langchain-core"
            )

        # Retrieve from ASMR
        result = self._pipeline.retrieve(
            query=query,
            top_k=self._top_k,
            require_reasoning=self._require_reasoning,
        )

        # Convert to LangChain Documents
        documents = []
        for memory in result.memories:
            doc = Document(
                page_content=memory.content,
                metadata={
                    "id": memory.id,
                    "source": memory.source,
                    "timestamp": memory.timestamp.isoformat(),
                    "tags": memory.tags,
                    **memory.metadata,
                },
            )
            documents.append(doc)

        return documents

    async def _aget_relevant_documents(self, query: str) -> list:
        """Async version of get_relevant_documents.

        Args:
            query: The search query.

        Returns:
            List of LangChain Document objects.
        """
        try:
            from langchain_core.documents import Document
        except ImportError:
            raise ImportError(
                "langchain-core is required. Install with: pip install langchain-core"
            )

        # Async retrieve from ASMR
        result = await self._pipeline.aretrieve(
            query=query,
            top_k=self._top_k,
            require_reasoning=self._require_reasoning,
        )

        # Convert to LangChain Documents
        documents = []
        for memory in result.memories:
            doc = Document(
                page_content=memory.content,
                metadata={
                    "id": memory.id,
                    "source": memory.source,
                    "timestamp": memory.timestamp.isoformat(),
                    "tags": memory.tags,
                    **memory.metadata,
                },
            )
            documents.append(doc)

        return documents

    # LangChain interface compatibility
    def get_relevant_documents(self, query: str) -> list:
        """LangChain interface method."""
        return self._get_relevant_documents(query)

    async def aget_relevant_documents(self, query: str) -> list:
        """LangChain async interface method."""
        return await self._aget_relevant_documents(query)

    def invoke(self, input: str, config: Optional[dict] = None) -> list:
        """LangChain Runnable interface."""
        return self._get_relevant_documents(input)

    async def ainvoke(self, input: str, config: Optional[dict] = None) -> list:
        """LangChain async Runnable interface."""
        return await self._aget_relevant_documents(input)

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

    def add_documents(self, documents: list) -> list[str]:
        """Add LangChain Documents to the memory store.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List of created memory IDs.
        """
        ids = []
        for doc in documents:
            metadata = doc.metadata.copy() if hasattr(doc, "metadata") else {}
            source = metadata.pop("source", "langchain")
            tags = metadata.pop("tags", [])

            memory = self._pipeline.add_memory(
                content=doc.page_content,
                source=source,
                metadata=metadata,
                tags=tags,
            )
            ids.append(memory.id)

        return ids

    @property
    def pipeline(self) -> RetrievalPipeline:
        """Get the underlying ASMR pipeline."""
        return self._pipeline


def create_asmr_retriever(
    top_k: int = 5,
    require_reasoning: bool = True,
    **kwargs: Any,
) -> ASMRLangChainRetriever:
    """Factory function to create an ASMR LangChain retriever.

    Args:
        top_k: Number of documents to retrieve.
        require_reasoning: Whether to use agent reasoning.
        **kwargs: Additional arguments for RetrievalPipeline.

    Returns:
        Configured ASMRLangChainRetriever.
    """
    return ASMRLangChainRetriever(
        top_k=top_k,
        require_reasoning=require_reasoning,
        **kwargs,
    )
