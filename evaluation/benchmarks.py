"""Benchmark runner for comparing ASMR vs baselines."""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from evaluation.baselines import BaselineRetriever, NaiveRAGRetriever, MMRRetriever, TimeWeightedRetriever
from evaluation.datasets import EvalScenario, StalenessDataset, ConflictDataset, MultiHopDataset
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    mean_reciprocal_rank,
    staleness_detection_rate,
    latency_comparison,
)
from retrieval.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results for a single benchmark run."""

    method: str
    dataset: str
    precision: float
    recall: float
    f1: float
    mrr: float
    latency_ms: float
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    results: list[BenchmarkResult]
    summary: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Generate markdown table of results.

        Returns:
            Markdown formatted table.
        """
        lines = [
            "# ASMR Benchmark Results\n",
            "| Method | Dataset | P@5 | R@5 | F1@5 | MRR | Latency (ms) |",
            "|--------|---------|-----|-----|------|-----|--------------|",
        ]

        for r in self.results:
            lines.append(
                f"| {r.method} | {r.dataset} | {r.precision:.3f} | {r.recall:.3f} | "
                f"{r.f1:.3f} | {r.mrr:.3f} | {r.latency_ms:.1f} |"
            )

        if self.summary:
            lines.append("\n## Summary\n")
            for key, value in self.summary.items():
                lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Dict representation.
        """
        return {
            "results": [
                {
                    "method": r.method,
                    "dataset": r.dataset,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                    "mrr": r.mrr,
                    "latency_ms": r.latency_ms,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
            "summary": self.summary,
        }


class BenchmarkRunner:
    """Runs benchmarks comparing ASMR against baselines."""

    def __init__(
        self,
        pipeline: Optional[RetrievalPipeline] = None,
        baselines: Optional[dict[str, BaselineRetriever]] = None,
        top_k: int = 5,
    ):
        """Initialize the benchmark runner.

        Args:
            pipeline: ASMR retrieval pipeline.
            baselines: Dict of baseline name to retriever.
            top_k: Number of results to retrieve.
        """
        self.pipeline = pipeline
        self.baselines = baselines or {
            "NaiveRAG": NaiveRAGRetriever(),
            "MMR": MMRRetriever(lambda_param=0.7),
            "TimeWeighted": TimeWeightedRetriever(half_life_days=30),
        }
        self.top_k = top_k

    def run_all(
        self,
        datasets: Optional[dict[str, list[EvalScenario]]] = None,
    ) -> BenchmarkReport:
        """Run all benchmarks.

        Args:
            datasets: Dict of dataset name to scenarios.
                     If None, generates default datasets.

        Returns:
            BenchmarkReport with all results.
        """
        if datasets is None:
            datasets = self._generate_default_datasets()

        results = []
        all_latencies = {"ASMR": [], "baselines": []}

        for dataset_name, scenarios in datasets.items():
            logger.info(f"Running benchmark on {dataset_name} ({len(scenarios)} scenarios)")

            # Run ASMR
            if self.pipeline:
                asmr_result = self._run_asmr(dataset_name, scenarios)
                results.append(asmr_result)
                all_latencies["ASMR"].append(asmr_result.latency_ms)

            # Run baselines
            for baseline_name, retriever in self.baselines.items():
                baseline_result = self._run_baseline(
                    baseline_name, dataset_name, scenarios, retriever
                )
                results.append(baseline_result)
                all_latencies["baselines"].append(baseline_result.latency_ms)

        # Compute summary
        summary = self._compute_summary(results, all_latencies)

        return BenchmarkReport(results=results, summary=summary)

    def _generate_default_datasets(self) -> dict[str, list[EvalScenario]]:
        """Generate default evaluation datasets.

        Returns:
            Dict of dataset name to scenarios.
        """
        return {
            "Staleness": StalenessDataset().generate(),
            "Conflict": ConflictDataset().generate(),
            "MultiHop": MultiHopDataset().generate(),
        }

    def _run_asmr(
        self,
        dataset_name: str,
        scenarios: list[EvalScenario],
    ) -> BenchmarkResult:
        """Run ASMR on a dataset.

        Args:
            dataset_name: Name of the dataset.
            scenarios: Evaluation scenarios.

        Returns:
            BenchmarkResult for ASMR.
        """
        all_retrieved = []
        all_relevant = []
        total_latency = 0.0

        for scenario in scenarios:
            # Add memories to pipeline
            for mem in scenario.memories:
                try:
                    self.pipeline.add_memory(
                        content=mem.content,
                        source=mem.source,
                        timestamp=mem.timestamp,
                    )
                except Exception:
                    pass  # Memory might already exist

            # Retrieve
            start = time.time()
            result = self.pipeline.retrieve(
                query=scenario.query,
                top_k=self.top_k,
                require_reasoning=True,
            )
            latency = (time.time() - start) * 1000
            total_latency += latency

            retrieved_ids = [m.id for m in result.memories]
            all_retrieved.append(retrieved_ids)
            all_relevant.append(set(scenario.ground_truth_ids))

        # Compute metrics
        avg_precision = sum(
            precision_at_k(r, rel, self.top_k)
            for r, rel in zip(all_retrieved, all_relevant)
        ) / len(scenarios)

        avg_recall = sum(
            recall_at_k(r, rel, self.top_k)
            for r, rel in zip(all_retrieved, all_relevant)
        ) / len(scenarios)

        avg_f1 = sum(
            f1_at_k(r, rel, self.top_k)
            for r, rel in zip(all_retrieved, all_relevant)
        ) / len(scenarios)

        mrr = mean_reciprocal_rank(all_retrieved, all_relevant)

        return BenchmarkResult(
            method="ASMR",
            dataset=dataset_name,
            precision=avg_precision,
            recall=avg_recall,
            f1=avg_f1,
            mrr=mrr,
            latency_ms=total_latency / len(scenarios),
        )

    def _run_baseline(
        self,
        baseline_name: str,
        dataset_name: str,
        scenarios: list[EvalScenario],
        retriever: BaselineRetriever,
    ) -> BenchmarkResult:
        """Run a baseline on a dataset.

        Args:
            baseline_name: Name of the baseline.
            dataset_name: Name of the dataset.
            scenarios: Evaluation scenarios.
            retriever: Baseline retriever.

        Returns:
            BenchmarkResult for the baseline.
        """
        all_retrieved = []
        all_relevant = []
        total_latency = 0.0

        for scenario in scenarios:
            # Create mock query embedding
            query_embedding = [0.1] * 384  # Mock embedding

            # Retrieve
            start = time.time()
            results = retriever.retrieve(
                query_embedding=query_embedding,
                memories=scenario.memories,
                top_k=self.top_k,
            )
            latency = (time.time() - start) * 1000
            total_latency += latency

            retrieved_ids = [m.id for m in results]
            all_retrieved.append(retrieved_ids)
            all_relevant.append(set(scenario.ground_truth_ids))

        # Compute metrics
        avg_precision = sum(
            precision_at_k(r, rel, self.top_k)
            for r, rel in zip(all_retrieved, all_relevant)
        ) / max(len(scenarios), 1)

        avg_recall = sum(
            recall_at_k(r, rel, self.top_k)
            for r, rel in zip(all_retrieved, all_relevant)
        ) / max(len(scenarios), 1)

        avg_f1 = sum(
            f1_at_k(r, rel, self.top_k)
            for r, rel in zip(all_retrieved, all_relevant)
        ) / max(len(scenarios), 1)

        mrr = mean_reciprocal_rank(all_retrieved, all_relevant)

        return BenchmarkResult(
            method=baseline_name,
            dataset=dataset_name,
            precision=avg_precision,
            recall=avg_recall,
            f1=avg_f1,
            mrr=mrr,
            latency_ms=total_latency / max(len(scenarios), 1),
        )

    def _compute_summary(
        self,
        results: list[BenchmarkResult],
        latencies: dict[str, list[float]],
    ) -> dict:
        """Compute summary statistics.

        Args:
            results: All benchmark results.
            latencies: Latency lists by method.

        Returns:
            Summary dict.
        """
        asmr_results = [r for r in results if r.method == "ASMR"]
        baseline_results = [r for r in results if r.method != "ASMR"]

        summary = {}

        if asmr_results:
            summary["ASMR Average F1"] = sum(r.f1 for r in asmr_results) / len(asmr_results)

        if baseline_results:
            best_baseline = max(baseline_results, key=lambda r: r.f1)
            summary["Best Baseline"] = f"{best_baseline.method} (F1: {best_baseline.f1:.3f})"

        if asmr_results and baseline_results:
            asmr_f1 = summary.get("ASMR Average F1", 0)
            baseline_f1 = sum(r.f1 for r in baseline_results) / len(baseline_results)
            improvement = ((asmr_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
            summary["F1 Improvement"] = f"{improvement:.1f}%"

        if latencies["ASMR"] and latencies["baselines"]:
            lat_comp = latency_comparison(latencies["ASMR"], latencies["baselines"])
            summary["Latency Overhead"] = f"{lat_comp['overhead_ratio']:.1f}x"

        return summary
