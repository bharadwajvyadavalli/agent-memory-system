#!/usr/bin/env python3
"""Research Assistant Example.

This example demonstrates how ASMR handles evolving knowledge over time,
such as research papers with findings that get updated or contradicted.
"""

from datetime import datetime, timedelta

from retrieval.pipeline import RetrievalPipeline


def main():
    print("=" * 60)
    print("ASMR Research Assistant Example")
    print("Handling evolving knowledge over time")
    print("=" * 60)

    # Initialize pipeline
    pipeline = RetrievalPipeline()

    # Simulate research papers over 6 months
    now = datetime.utcnow()

    # Initial study (6 months ago)
    pipeline.add_memory(
        content="Study finds that technique A improves model accuracy by 15%. "
                "Tested on benchmark dataset X with 1000 samples.",
        source="paper_smith_2023",
        timestamp=now - timedelta(days=180),
        metadata={"authors": "Smith et al.", "venue": "ICML 2023"},
        tags=["ml", "accuracy", "technique-a"],
    )

    # Follow-up study with nuance (4 months ago)
    pipeline.add_memory(
        content="Technique A shows 15% improvement on small datasets, but only 5% "
                "improvement on large-scale datasets. Domain adaptation required.",
        source="paper_jones_2023",
        timestamp=now - timedelta(days=120),
        metadata={"authors": "Jones et al.", "venue": "NeurIPS 2023"},
        tags=["ml", "accuracy", "technique-a", "scale"],
    )

    # Contradicting study (2 months ago)
    pipeline.add_memory(
        content="Our experiments show technique A provides minimal improvement (2-3%) "
                "when proper baselines are used. Previous studies had flawed methodology.",
        source="paper_lee_2024",
        timestamp=now - timedelta(days=60),
        metadata={"authors": "Lee et al.", "venue": "ICLR 2024"},
        tags=["ml", "accuracy", "technique-a", "methodology"],
    )

    # New technique (1 month ago)
    pipeline.add_memory(
        content="Introducing technique B: achieves 20% accuracy improvement over baselines "
                "across all dataset sizes. Technique A is now considered outdated.",
        source="paper_chen_2024",
        timestamp=now - timedelta(days=30),
        metadata={"authors": "Chen et al.", "venue": "CVPR 2024"},
        tags=["ml", "accuracy", "technique-b"],
    )

    # Meta-analysis (recent)
    pipeline.add_memory(
        content="Meta-analysis of 50 papers: Technique B is the current state-of-the-art "
                "with mean improvement of 18%. Technique A improvements were overstated.",
        source="paper_survey_2024",
        timestamp=now - timedelta(days=7),
        metadata={"authors": "Survey Team", "venue": "ACM Computing Surveys"},
        tags=["ml", "survey", "technique-a", "technique-b"],
    )

    print(f"\nLoaded {len(pipeline)} research papers.\n")

    # Query about current best techniques
    queries = [
        "What is the best technique for improving model accuracy?",
        "How effective is technique A?",
        "What do recent studies say about accuracy improvements?",
    ]

    for query in queries:
        print("=" * 60)
        print(f"Query: '{query}'")
        print("=" * 60)

        result = pipeline.retrieve(query=query, top_k=3, require_reasoning=True)

        print("\nAgent Analysis:")
        for decision in result.agent_decisions[-5:]:  # Show last 5 decisions
            print(f"  [{decision.agent_name}] {decision.action} "
                  f"(conf: {decision.confidence:.2f})")

        print("\nCurated Response:")
        print(result.final_context[:500] + "..." if len(result.final_context) > 500 else result.final_context)

        print(f"\nMetrics: {result.metadata.get('latency_ms', 0):.0f}ms, "
              f"{len(result.memories)} memories used\n")


if __name__ == "__main__":
    main()
