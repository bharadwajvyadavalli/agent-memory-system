"""Evaluation metrics for memory retrieval."""

from typing import Optional

from memory.schema import Memory, ConflictRecord


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: Optional[int] = None,
) -> float:
    """Calculate precision@k.

    Args:
        retrieved_ids: List of retrieved memory IDs in order.
        relevant_ids: Set of ground truth relevant IDs.
        k: Number of top results to consider. None = all.

    Returns:
        Precision score (0-1).
    """
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    if not retrieved_ids:
        return 0.0

    relevant_retrieved = sum(1 for id in retrieved_ids if id in relevant_ids)
    return relevant_retrieved / len(retrieved_ids)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: Optional[int] = None,
) -> float:
    """Calculate recall@k.

    Args:
        retrieved_ids: List of retrieved memory IDs in order.
        relevant_ids: Set of ground truth relevant IDs.
        k: Number of top results to consider. None = all.

    Returns:
        Recall score (0-1).
    """
    if not relevant_ids:
        return 0.0

    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    relevant_retrieved = sum(1 for id in retrieved_ids if id in relevant_ids)
    return relevant_retrieved / len(relevant_ids)


def f1_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: Optional[int] = None,
) -> float:
    """Calculate F1@k.

    Args:
        retrieved_ids: List of retrieved memory IDs.
        relevant_ids: Set of ground truth relevant IDs.
        k: Number of top results to consider.

    Returns:
        F1 score (0-1).
    """
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


def mean_reciprocal_rank(
    retrieved_ids_list: list[list[str]],
    relevant_ids_list: list[set[str]],
) -> float:
    """Calculate Mean Reciprocal Rank across multiple queries.

    Args:
        retrieved_ids_list: List of retrieved ID lists (one per query).
        relevant_ids_list: List of relevant ID sets (one per query).

    Returns:
        MRR score (0-1).
    """
    if not retrieved_ids_list:
        return 0.0

    reciprocal_ranks = []

    for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
        rr = 0.0
        for i, id in enumerate(retrieved, 1):
            if id in relevant:
                rr = 1.0 / i
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def conflict_resolution_accuracy(
    records: list[ConflictRecord],
    ground_truth_winners: dict[tuple[str, str], str],
) -> float:
    """Calculate accuracy of conflict resolution.

    Args:
        records: List of ConflictRecord from ConflictAgent.
        ground_truth_winners: Dict mapping (id_a, id_b) to correct winner_id.

    Returns:
        Accuracy score (0-1).
    """
    if not records or not ground_truth_winners:
        return 0.0

    correct = 0
    total = 0

    for record in records:
        # Check both orderings of the pair
        key = (record.memory_a_id, record.memory_b_id)
        alt_key = (record.memory_b_id, record.memory_a_id)

        truth = ground_truth_winners.get(key) or ground_truth_winners.get(alt_key)
        if truth is not None:
            total += 1
            if record.winner_id == truth:
                correct += 1

    return correct / total if total > 0 else 0.0


def staleness_detection_rate(
    retrieved_memories: list[Memory],
    stale_ids: set[str],
) -> dict[str, float]:
    """Calculate staleness detection metrics.

    Args:
        retrieved_memories: Memories returned by retrieval.
        stale_ids: Set of IDs known to be stale.

    Returns:
        Dict with detection_rate, false_inclusion_rate.
    """
    retrieved_ids = {m.id for m in retrieved_memories}

    # Stale memories that were NOT returned (correctly filtered)
    correctly_filtered = len(stale_ids - retrieved_ids)

    # Stale memories that WERE returned (incorrectly included)
    false_inclusions = len(stale_ids & retrieved_ids)

    total_stale = len(stale_ids)

    return {
        "detection_rate": correctly_filtered / total_stale if total_stale > 0 else 1.0,
        "false_inclusion_rate": false_inclusions / total_stale if total_stale > 0 else 0.0,
        "stale_in_results": false_inclusions,
        "total_stale": total_stale,
    }


def latency_comparison(
    asmr_latencies: list[float],
    baseline_latencies: list[float],
) -> dict[str, dict[str, float]]:
    """Compare latency statistics between ASMR and baseline.

    Args:
        asmr_latencies: List of ASMR latencies in ms.
        baseline_latencies: List of baseline latencies in ms.

    Returns:
        Dict with statistics for each method.
    """
    import statistics

    def compute_stats(latencies: list[float]) -> dict[str, float]:
        if not latencies:
            return {"mean": 0, "p50": 0, "p95": 0, "p99": 0}

        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        return {
            "mean": statistics.mean(latencies),
            "p50": sorted_lat[int(n * 0.5)],
            "p95": sorted_lat[int(n * 0.95)] if n > 1 else sorted_lat[-1],
            "p99": sorted_lat[int(n * 0.99)] if n > 1 else sorted_lat[-1],
            "min": min(latencies),
            "max": max(latencies),
        }

    return {
        "asmr": compute_stats(asmr_latencies),
        "baseline": compute_stats(baseline_latencies),
        "overhead_ratio": (
            statistics.mean(asmr_latencies) / statistics.mean(baseline_latencies)
            if baseline_latencies and asmr_latencies
            else 0.0
        ),
    }


def ndcg_at_k(
    retrieved_ids: list[str],
    relevance_scores: dict[str, float],
    k: Optional[int] = None,
) -> float:
    """Calculate NDCG@k (Normalized Discounted Cumulative Gain).

    Args:
        retrieved_ids: List of retrieved memory IDs in order.
        relevance_scores: Dict mapping ID to relevance score (0-1).
        k: Number of top results to consider.

    Returns:
        NDCG score (0-1).
    """
    import math

    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    if not retrieved_ids:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for i, id in enumerate(retrieved_ids, 1):
        rel = relevance_scores.get(id, 0.0)
        dcg += (2 ** rel - 1) / math.log2(i + 1)

    # Calculate ideal DCG
    ideal_scores = sorted(relevance_scores.values(), reverse=True)
    if k is not None:
        ideal_scores = ideal_scores[:k]

    idcg = 0.0
    for i, rel in enumerate(ideal_scores, 1):
        idcg += (2 ** rel - 1) / math.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg
