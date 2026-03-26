#!/usr/bin/env python3
"""Customer Support Example.

This example demonstrates how ASMR handles policy changes in a customer
support context, ensuring agents always return the most current policies.
"""

from datetime import datetime, timedelta

from retrieval.pipeline import RetrievalPipeline


def main():
    print("=" * 60)
    print("ASMR Customer Support Example")
    print("Handling policy changes over time")
    print("=" * 60)

    # Initialize pipeline
    pipeline = RetrievalPipeline()

    now = datetime.utcnow()

    # === RETURN POLICY EVOLUTION ===
    pipeline.add_memory(
        content="Return Policy: 60 days full refund on all items.",
        source="policy_2021",
        timestamp=now - timedelta(days=1000),
        tags=["policy", "returns"],
    )

    pipeline.add_memory(
        content="Return Policy Update: 30 days full refund. After 30 days, "
                "store credit only. Effective March 2023.",
        source="policy_2023",
        timestamp=now - timedelta(days=365),
        tags=["policy", "returns"],
    )

    pipeline.add_memory(
        content="CURRENT Return Policy (January 2024): 15 days full refund. "
                "No returns after 15 days except for defective items.",
        source="policy_2024",
        timestamp=now - timedelta(days=60),
        tags=["policy", "returns"],
    )

    # === SHIPPING INFORMATION ===
    pipeline.add_memory(
        content="Free shipping on orders over $50. Standard delivery 5-7 business days.",
        source="shipping_old",
        timestamp=now - timedelta(days=400),
        tags=["shipping"],
    )

    pipeline.add_memory(
        content="Updated shipping: Free shipping on orders over $35. "
                "Express delivery now available (2-3 days) for $9.99.",
        source="shipping_2024",
        timestamp=now - timedelta(days=30),
        tags=["shipping"],
    )

    # === WARRANTY ===
    pipeline.add_memory(
        content="All electronics come with 1-year manufacturer warranty.",
        source="warranty_standard",
        timestamp=now - timedelta(days=500),
        tags=["warranty", "electronics"],
    )

    pipeline.add_memory(
        content="Extended warranty available: 3 years for $49.99. "
                "Covers accidental damage. Note: Standard warranty is still 1 year.",
        source="warranty_extended",
        timestamp=now - timedelta(days=100),
        tags=["warranty", "electronics"],
    )

    # === SUPPORT HOURS ===
    pipeline.add_memory(
        content="Customer support available Monday-Friday 9 AM - 5 PM EST.",
        source="support_old",
        timestamp=now - timedelta(days=600),
        tags=["support", "hours"],
    )

    pipeline.add_memory(
        content="New support hours: 24/7 chat support available. "
                "Phone support: Monday-Saturday 8 AM - 8 PM EST.",
        source="support_2024",
        timestamp=now - timedelta(days=45),
        tags=["support", "hours"],
    )

    print(f"\nLoaded {len(pipeline)} support documents.\n")

    # Simulate customer queries
    customer_queries = [
        ("What is your return policy?", "return"),
        ("How long do I have to return something?", "return"),
        ("Is shipping free?", "shipping"),
        ("What warranty comes with my purchase?", "warranty"),
        ("When is customer support available?", "support"),
    ]

    for query, topic in customer_queries:
        print("=" * 60)
        print(f"Customer: '{query}'")
        print("=" * 60)

        result = pipeline.retrieve(query=query, top_k=3, require_reasoning=True)

        # Show reasoning for educational purposes
        print("\nASMR Agent Reasoning:")
        for decision in result.agent_decisions:
            if decision.action in ["discard", "flag_conflict"]:
                print(f"  - Filtered: {decision.reasoning[:60]}...")

        print("\nSupport Agent Response:")
        print(f"  {result.final_context}")

        # Verify we got current info
        for mem in result.memories:
            if "old" in mem.source.lower() or "2021" in mem.source or "2023" in mem.source:
                print(f"  ⚠️  WARNING: Old document included: {mem.source}")
            else:
                print(f"  ✓ Current document: {mem.source}")

        print()


if __name__ == "__main__":
    main()
