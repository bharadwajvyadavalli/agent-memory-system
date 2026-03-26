"""Pydantic v2 data models for ASMR memory system."""

from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Memory(BaseModel):
    """A single memory unit in the ASMR system."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., description="The textual content of the memory")
    embedding: Optional[list[float]] = Field(
        default=None, description="Vector embedding of the content"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this memory was created",
    )
    source: str = Field(..., description="Origin of this memory (e.g., document name, URL)")
    version: int = Field(default=1, description="Version number for tracking updates")
    supersedes: Optional[str] = Field(
        default=None,
        description="UUID of the memory this one replaces",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (e.g., author, category)",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization and filtering",
    )
    access_count: int = Field(default=0, description="Number of times this memory was retrieved")
    last_accessed: Optional[datetime] = Field(
        default=None,
        description="Last time this memory was retrieved",
    )
    is_active: bool = Field(
        default=True,
        description="Whether this memory is active (False = soft deleted or superseded)",
    )


class MemoryQuery(BaseModel):
    """Query parameters for memory retrieval."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = Field(..., description="The search query")
    filters: Optional[dict] = Field(
        default=None,
        description="Optional filters (time_range, source, tags)",
    )
    top_k: int = Field(default=5, description="Number of memories to retrieve")
    require_reasoning: bool = Field(
        default=True,
        description="Whether to use agent reasoning or just embedding similarity",
    )


class AgentDecision(BaseModel):
    """A decision made by an agent about a memory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_name: str = Field(..., description="Name of the agent making the decision")
    memory_id: str = Field(..., description="ID of the memory being evaluated")
    action: Literal["keep", "discard", "merge", "flag_conflict"] = Field(
        ..., description="The action to take"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for this decision"
    )
    reasoning: str = Field(..., description="Chain-of-thought reasoning for the decision")


class RetrievalResult(BaseModel):
    """Result of a memory retrieval operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memories: list[Memory] = Field(
        default_factory=list,
        description="Retrieved memories after agent processing",
    )
    agent_decisions: list[AgentDecision] = Field(
        default_factory=list,
        description="All decisions made by agents during retrieval",
    )
    final_context: str = Field(
        default="",
        description="Synthesized context string for use in prompts",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Retrieval metadata (latency_ms, candidates_considered, etc.)",
    )


class ConflictRecord(BaseModel):
    """Record of a detected conflict between memories."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory_a_id: str = Field(..., description="ID of the first memory in conflict")
    memory_b_id: str = Field(..., description="ID of the second memory in conflict")
    conflict_type: Literal["full_contradiction", "partial_update", "scope_difference"] = Field(
        ..., description="Type of conflict detected"
    )
    winner_id: Optional[str] = Field(
        default=None,
        description="ID of the memory that should be preferred",
    )
    reasoning: str = Field(..., description="Explanation of the conflict and resolution")


class TimeRange(BaseModel):
    """Time range filter for queries."""

    start: Optional[datetime] = None
    end: Optional[datetime] = None


class MemoryFilter(BaseModel):
    """Structured filters for memory queries."""

    time_range: Optional[TimeRange] = None
    sources: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    require_active: bool = True
