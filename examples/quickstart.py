#!/usr/bin/env python3
"""ASMR Quickstart Example.

This example demonstrates the basic usage of ASMR:
1. Create a retrieval pipeline
2. Add memories (some stale, some conflicting)
3. Query and see agent reasoning
"""

from retrieval.pipeline import RetrievalPipeline


def main():
    # Initialize the ASMR pipeline
    print("Initializing ASMR pipeline...")
    pipeline = RetrievalPipeline()

    # Add memories with temporal and conflicting information
    print("\nAdding memories...")

    # Old policy (should be superseded)
    pipeline.add_memory(
        content="Return policy: 30 days full refund, no questions asked.",
        source="policy_v1",
        metadata={"effective_date": "2022-01-01", "version": 1},
        tags=["policy", "returns"],
    )

    # New policy (should be preferred)
    pipeline.add_memory(
        content="Updated: Return policy is now 15 days. Effective January 2024.",
        source="policy_v2",
        metadata={"effective_date": "2024-01-01", "version": 2},
        tags=["policy", "returns"],
    )

    # Irrelevant but potentially matching content
    pipeline.add_memory(
        content="Our store sells electronics and home appliances.",
        source="about",
        tags=["general", "products"],
    )

    # Another piece of relevant info
    pipeline.add_memory(
        content="All returns require original receipt and packaging.",
        source="policy_v2",
        metadata={"effective_date": "2024-01-01"},
        tags=["policy", "returns"],
    )

    print(f"Added {len(pipeline)} memories to the store.\n")

    # Query with agent reasoning
    query = "What is the current return policy?"
    print(f"Query: '{query}'\n")
    print("Running ASMR retrieval with agent reasoning...\n")

    result = pipeline.retrieve(
        query=query,
        top_k=5,
        require_reasoning=True,
    )

    # Display results
    print("=" * 60)
    print("AGENT DECISIONS:")
    print("=" * 60)

    for decision in result.agent_decisions:
        print(f"\n[{decision.agent_name}] Memory: {decision.memory_id[:8]}...")
        print(f"  Action: {decision.action}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Reasoning: {decision.reasoning[:100]}...")

    print("\n" + "=" * 60)
    print("FINAL CURATED CONTEXT:")
    print("=" * 60)
    print(result.final_context)

    print("\n" + "=" * 60)
    print("METADATA:")
    print("=" * 60)
    print(f"  Latency: {result.metadata.get('latency_ms', 0):.1f}ms")
    print(f"  Candidates considered: {result.metadata.get('candidates_considered', 0)}")
    print(f"  Candidates filtered: {result.metadata.get('candidates_filtered', 0)}")
    print(f"  Agents used: {result.metadata.get('agents_used', [])}")

    # Fast retrieval comparison
    print("\n" + "=" * 60)
    print("FAST RETRIEVAL (no agent reasoning):")
    print("=" * 60)

    fast_result = pipeline.retrieve_fast(query=query, top_k=3)
    for mem in fast_result.memories:
        print(f"  - {mem.content[:80]}...")

    print(f"\nFast latency: {fast_result.metadata.get('latency_ms', 0):.1f}ms")


if __name__ == "__main__":
    main()
