"""Synthetic evaluation datasets."""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

from memory.schema import Memory


@dataclass
class EvalScenario:
    """A single evaluation scenario."""

    id: str
    name: str
    description: str
    memories: list[Memory]
    query: str
    ground_truth_ids: list[str]  # IDs of correct memories to return
    metadata: dict = field(default_factory=dict)


class StalenessDataset:
    """Dataset for evaluating staleness detection."""

    SCENARIOS = [
        {
            "name": "CEO Change",
            "memories": [
                ("John Smith is the CEO of TechCorp.", "company_info", -365),  # 1 year ago
                ("Jane Doe was appointed CEO of TechCorp in March 2024.", "company_news", -30),  # 30 days ago
            ],
            "query": "Who is the CEO of TechCorp?",
            "correct": [1],  # Index of correct memory
        },
        {
            "name": "Policy Update",
            "memories": [
                ("Return policy: 30 days full refund, no questions asked.", "policy_v1", -180),
                ("Updated: Return policy changed to 15 days. Effective January 2024.", "policy_v2", -30),
            ],
            "query": "What is the return policy?",
            "correct": [1],
        },
        {
            "name": "Product Version",
            "memories": [
                ("Product X version 2.0 released with new features.", "release_notes", -365),
                ("Product X version 2.5 introduces performance improvements.", "release_notes", -180),
                ("Product X version 3.0 is our latest release with AI features.", "release_notes", -7),
            ],
            "query": "What is the latest version of Product X?",
            "correct": [2],
        },
        {
            "name": "Pricing Change",
            "memories": [
                ("Enterprise plan: $99/month per user.", "pricing", -200),
                ("Price update: Enterprise plan now $129/month per user.", "pricing", -60),
            ],
            "query": "How much is the enterprise plan?",
            "correct": [1],
        },
        {
            "name": "Office Location",
            "memories": [
                ("Our headquarters is located at 123 Main St, San Francisco.", "about", -500),
                ("We've moved! New headquarters at 456 Tech Blvd, Palo Alto.", "about", -30),
            ],
            "query": "Where is your office located?",
            "correct": [1],
        },
    ]

    def generate(self, n_scenarios: Optional[int] = None) -> list[EvalScenario]:
        """Generate staleness evaluation scenarios.

        Args:
            n_scenarios: Number of scenarios to generate. None = all.

        Returns:
            List of EvalScenario objects.
        """
        scenarios = self.SCENARIOS[:n_scenarios] if n_scenarios else self.SCENARIOS
        results = []

        for scenario in scenarios:
            memories = []
            ground_truth = []
            now = datetime.utcnow()

            for i, (content, source, days_ago) in enumerate(scenario["memories"]):
                mem = Memory(
                    id=str(uuid4()),
                    content=content,
                    source=source,
                    timestamp=now + timedelta(days=days_ago),
                    embedding=[random.random() for _ in range(384)],  # Mock embedding
                )
                memories.append(mem)

                if i in scenario["correct"]:
                    ground_truth.append(mem.id)

            results.append(EvalScenario(
                id=str(uuid4()),
                name=scenario["name"],
                description=f"Staleness test: {scenario['name']}",
                memories=memories,
                query=scenario["query"],
                ground_truth_ids=ground_truth,
                metadata={"type": "staleness"},
            ))

        return results


class ConflictDataset:
    """Dataset for evaluating conflict detection and resolution."""

    SCENARIOS = [
        {
            "name": "Full Contradiction - CEO",
            "memories": [
                ("The CEO of Acme Corp is Robert Brown.", "annual_report_2022", -400),
                ("Sarah Johnson was appointed CEO of Acme Corp.", "press_release", -60),
            ],
            "query": "Who is the CEO of Acme Corp?",
            "conflict_type": "full_contradiction",
            "winner": 1,
        },
        {
            "name": "Partial Update - Working Hours",
            "memories": [
                ("Office hours are 9 AM to 5 PM, Monday through Friday.", "employee_handbook", -365),
                ("Remote employees can work flexible hours between 7 AM and 7 PM.", "remote_policy", -60),
            ],
            "query": "What are the working hours?",
            "conflict_type": "partial_update",
            "winner": None,  # Both valid for different contexts
        },
        {
            "name": "Scope Difference - Budget",
            "memories": [
                ("Company annual budget is $10 million.", "finance", -30),
                ("Engineering department budget is $3 million.", "finance", -30),
            ],
            "query": "What is the budget?",
            "conflict_type": "scope_difference",
            "winner": None,  # Both valid, different scopes
        },
        {
            "name": "Full Contradiction - Meeting Time",
            "memories": [
                ("Weekly standup is at 10 AM on Mondays.", "team_calendar", -100),
                ("Standup time changed to 9:30 AM starting this week.", "team_slack", -3),
            ],
            "query": "When is the weekly standup?",
            "conflict_type": "full_contradiction",
            "winner": 1,
        },
    ]

    def generate(self, n_scenarios: Optional[int] = None) -> list[EvalScenario]:
        """Generate conflict evaluation scenarios.

        Args:
            n_scenarios: Number of scenarios to generate.

        Returns:
            List of EvalScenario objects.
        """
        scenarios = self.SCENARIOS[:n_scenarios] if n_scenarios else self.SCENARIOS
        results = []

        for scenario in scenarios:
            memories = []
            ground_truth = []
            now = datetime.utcnow()

            for i, (content, source, days_ago) in enumerate(scenario["memories"]):
                mem = Memory(
                    id=str(uuid4()),
                    content=content,
                    source=source,
                    timestamp=now + timedelta(days=days_ago),
                    embedding=[random.random() for _ in range(384)],
                )
                memories.append(mem)

                # For conflicts, ground truth depends on type
                if scenario["winner"] is not None:
                    if i == scenario["winner"]:
                        ground_truth.append(mem.id)
                else:
                    # Both are valid
                    ground_truth.append(mem.id)

            results.append(EvalScenario(
                id=str(uuid4()),
                name=scenario["name"],
                description=f"Conflict test: {scenario['name']}",
                memories=memories,
                query=scenario["query"],
                ground_truth_ids=ground_truth,
                metadata={
                    "type": "conflict",
                    "conflict_type": scenario["conflict_type"],
                    "winner_index": scenario["winner"],
                },
            ))

        return results


class MultiHopDataset:
    """Dataset for evaluating multi-hop reasoning."""

    SCENARIOS = [
        {
            "name": "Project Budget Query",
            "memories": [
                ("Sarah leads the Alpha project.", "org_chart", -30),
                ("Alpha project budget is $500,000.", "budgets", -30),
                ("Sarah was promoted to Senior Manager.", "hr_updates", -15),
            ],
            "query": "What is the budget for Sarah's project?",
            "correct": [0, 1],  # Need both to answer
            "reasoning": "Need to know Sarah leads Alpha, then look up Alpha's budget",
        },
        {
            "name": "Team Location Query",
            "memories": [
                ("John works in the Data Science team.", "team_roster", -30),
                ("Data Science team is based in Building C.", "office_map", -30),
                ("Building C has been renovated recently.", "facilities", -10),
            ],
            "query": "Where does John work?",
            "correct": [0, 1],
            "reasoning": "Need John's team, then team location",
        },
        {
            "name": "Manager Contact",
            "memories": [
                ("Mike reports to Lisa Chen.", "org_chart", -60),
                ("Lisa Chen's email is lisa.chen@company.com.", "directory", -30),
                ("Lisa is currently on vacation.", "out_of_office", -1),
            ],
            "query": "How do I contact Mike's manager?",
            "correct": [0, 1],
            "reasoning": "Need Mike's manager, then contact info",
        },
    ]

    def generate(self, n_scenarios: Optional[int] = None) -> list[EvalScenario]:
        """Generate multi-hop evaluation scenarios.

        Args:
            n_scenarios: Number of scenarios to generate.

        Returns:
            List of EvalScenario objects.
        """
        scenarios = self.SCENARIOS[:n_scenarios] if n_scenarios else self.SCENARIOS
        results = []

        for scenario in scenarios:
            memories = []
            ground_truth = []
            now = datetime.utcnow()

            for i, (content, source, days_ago) in enumerate(scenario["memories"]):
                mem = Memory(
                    id=str(uuid4()),
                    content=content,
                    source=source,
                    timestamp=now + timedelta(days=days_ago),
                    embedding=[random.random() for _ in range(384)],
                )
                memories.append(mem)

                if i in scenario["correct"]:
                    ground_truth.append(mem.id)

            results.append(EvalScenario(
                id=str(uuid4()),
                name=scenario["name"],
                description=f"Multi-hop test: {scenario['name']}",
                memories=memories,
                query=scenario["query"],
                ground_truth_ids=ground_truth,
                metadata={
                    "type": "multi_hop",
                    "reasoning": scenario["reasoning"],
                    "hops_required": len(scenario["correct"]),
                },
            ))

        return results


class RelevanceDataset:
    """Dataset for evaluating relevance detection (false positive filtering)."""

    SCENARIOS = [
        {
            "name": "Python Programming vs Comedy",
            "memories": [
                ("Python 3.12 introduces new performance improvements.", "docs", -30, True),
                ("Monty Python's Flying Circus aired from 1969 to 1974.", "wikipedia", -30, False),
                ("Python async/await syntax simplifies concurrent code.", "tutorial", -60, True),
            ],
            "query": "Python performance optimization",
        },
        {
            "name": "Apple Company vs Fruit",
            "memories": [
                ("Apple reported Q4 earnings of $90 billion.", "news", -7, True),
                ("Apple pie is a classic American dessert.", "recipes", -30, False),
                ("Apple's M3 chip shows 40% performance gain.", "tech", -14, True),
            ],
            "query": "Apple quarterly earnings",
        },
        {
            "name": "Java Programming vs Coffee",
            "memories": [
                ("Java 21 introduces virtual threads.", "docs", -30, True),
                ("Java coffee originated from Indonesia.", "food", -30, False),
                ("The JVM garbage collector has been optimized.", "tech", -60, True),
            ],
            "query": "Java memory management",
        },
    ]

    def generate(self, n_scenarios: Optional[int] = None) -> list[EvalScenario]:
        """Generate relevance evaluation scenarios.

        Args:
            n_scenarios: Number of scenarios to generate.

        Returns:
            List of EvalScenario objects.
        """
        scenarios = self.SCENARIOS[:n_scenarios] if n_scenarios else self.SCENARIOS
        results = []

        for scenario in scenarios:
            memories = []
            ground_truth = []
            now = datetime.utcnow()

            for content, source, days_ago, is_relevant in scenario["memories"]:
                mem = Memory(
                    id=str(uuid4()),
                    content=content,
                    source=source,
                    timestamp=now + timedelta(days=days_ago),
                    embedding=[random.random() for _ in range(384)],
                )
                memories.append(mem)

                if is_relevant:
                    ground_truth.append(mem.id)

            results.append(EvalScenario(
                id=str(uuid4()),
                name=scenario["name"],
                description=f"Relevance test: {scenario['name']}",
                memories=memories,
                query=scenario["query"],
                ground_truth_ids=ground_truth,
                metadata={"type": "relevance"},
            ))

        return results
