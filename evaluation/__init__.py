"""Evaluation module for ASMR - Benchmarks and metrics."""

from evaluation.benchmarks import BenchmarkRunner, BenchmarkReport
from evaluation.baselines import NaiveRAGRetriever, MMRRetriever, TimeWeightedRetriever
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    mean_reciprocal_rank,
    conflict_resolution_accuracy,
    staleness_detection_rate,
    latency_comparison,
)
from evaluation.datasets import (
    EvalScenario,
    StalenessDataset,
    ConflictDataset,
    MultiHopDataset,
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkReport",
    "NaiveRAGRetriever",
    "MMRRetriever",
    "TimeWeightedRetriever",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "mean_reciprocal_rank",
    "conflict_resolution_accuracy",
    "staleness_detection_rate",
    "latency_comparison",
    "EvalScenario",
    "StalenessDataset",
    "ConflictDataset",
    "MultiHopDataset",
]
