#!/usr/bin/env python3
"""Personal Assistant Example.

This example demonstrates how ASMR handles evolving user preferences,
resolving contradictions in personal information gracefully.
"""

from datetime import datetime, timedelta

from retrieval.pipeline import RetrievalPipeline


def main():
    print("=" * 60)
    print("ASMR Personal Assistant Example")
    print("Handling evolving user preferences")
    print("=" * 60)

    # Initialize pipeline
    pipeline = RetrievalPipeline()

    now = datetime.utcnow()

    # === FOOD PREFERENCES EVOLUTION ===
    pipeline.add_memory(
        content="User loves Italian food, especially pasta and pizza. "
                "Favorite restaurant is Luigi's.",
        source="preference_food_1",
        timestamp=now - timedelta(days=180),
        tags=["food", "preferences", "italian"],
    )

    pipeline.add_memory(
        content="User mentioned they're trying to cut back on carbs. "
                "Looking for low-carb alternatives.",
        source="preference_food_2",
        timestamp=now - timedelta(days=60),
        tags=["food", "preferences", "health"],
    )

    pipeline.add_memory(
        content="User started a keto diet. Avoiding bread, pasta, rice. "
                "Interested in protein-rich meals.",
        source="preference_food_3",
        timestamp=now - timedelta(days=14),
        tags=["food", "preferences", "keto", "health"],
    )

    # === SCHEDULE PREFERENCES ===
    pipeline.add_memory(
        content="User prefers morning meetings, usually starts work at 9 AM.",
        source="schedule_1",
        timestamp=now - timedelta(days=200),
        tags=["schedule", "preferences"],
    )

    pipeline.add_memory(
        content="User now works remotely with flexible hours. "
                "Prefers afternoon meetings after 2 PM.",
        source="schedule_2",
        timestamp=now - timedelta(days=30),
        tags=["schedule", "preferences", "remote"],
    )

    # === TRAVEL PREFERENCES ===
    pipeline.add_memory(
        content="User enjoys beach vacations. Favorite destination: Hawaii.",
        source="travel_1",
        timestamp=now - timedelta(days=365),
        tags=["travel", "preferences"],
    )

    pipeline.add_memory(
        content="User mentioned interest in mountain hiking and nature trips. "
                "Considering Colorado next.",
        source="travel_2",
        timestamp=now - timedelta(days=45),
        tags=["travel", "preferences", "hiking"],
    )

    # === COMMUNICATION PREFERENCES ===
    pipeline.add_memory(
        content="User prefers phone calls for important matters.",
        source="comm_1",
        timestamp=now - timedelta(days=300),
        tags=["communication", "preferences"],
    )

    pipeline.add_memory(
        content="User now prefers text messages and Slack. "
                "Phone calls only for emergencies.",
        source="comm_2",
        timestamp=now - timedelta(days=20),
        tags=["communication", "preferences"],
    )

    print(f"\nLoaded {len(pipeline)} user preference memories.\n")

    # Personal assistant queries
    queries = [
        "What food should I recommend for dinner tonight?",
        "When is a good time to schedule a meeting?",
        "What vacation should I suggest?",
        "How should I contact the user?",
        "What are the user's dietary restrictions?",
    ]

    for query in queries:
        print("=" * 60)
        print(f"Query: '{query}'")
        print("=" * 60)

        result = pipeline.retrieve(query=query, top_k=3, require_reasoning=True)

        # Show how ASMR handles preference evolution
        print("\nPreference Analysis:")

        kept_memories = []
        filtered_memories = []

        for decision in result.agent_decisions:
            mem = pipeline.get_memory(decision.memory_id)
            if mem:
                if decision.action in ["keep", "merge"]:
                    kept_memories.append((mem, decision))
                else:
                    filtered_memories.append((mem, decision))

        if filtered_memories:
            print("  Outdated preferences filtered:")
            for mem, dec in filtered_memories[:2]:
                print(f"    - {mem.content[:50]}...")
                print(f"      Reason: {dec.reasoning[:60]}...")

        if kept_memories:
            print("  Current preferences kept:")
            for mem, dec in kept_memories[:2]:
                print(f"    - {mem.content[:50]}...")

        print(f"\nAssistant Response:\n  {result.final_context}")
        print()


if __name__ == "__main__":
    main()
