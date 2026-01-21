"""
Ranking evaluation metrics.

Implements standard IR metrics:
- NDCG (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- Recall@K
- MRR (Mean Reciprocal Rank)
- Precision@K
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """
    Compute Discounted Cumulative Gain at K.
    
    DCG = Σ (2^rel_i - 1) / log2(i + 2) for i in [0, k)
    
    Args:
        relevance: Relevance scores in rank order
        k: Cutoff position
        
    Returns:
        DCG score
    """
    relevance = np.asarray(relevance)[:k]
    n = len(relevance)
    
    if n == 0:
        return 0.0
    
    # Gains
    gains = (2 ** relevance) - 1
    
    # Discounts
    discounts = np.log2(np.arange(2, n + 2))
    
    return float(np.sum(gains / discounts))


def ndcg_at_k(
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
    k: int = 10,
) -> float:
    """
    Compute Normalized DCG at K.
    
    NDCG = DCG / IDCG where IDCG is the ideal DCG.
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted scores (used for ranking)
        k: Cutoff position
        
    Returns:
        NDCG@K score in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Sort by predicted scores (descending)
    order = np.argsort(-y_pred)
    y_true_sorted = y_true[order]
    
    # Compute DCG for the predicted ranking
    dcg = dcg_at_k(y_true_sorted, k)
    
    # Compute ideal DCG (perfect ranking)
    ideal_order = np.argsort(-y_true)
    y_true_ideal = y_true[ideal_order]
    idcg = dcg_at_k(y_true_ideal, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def precision_at_k(
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
    k: int = 10,
    threshold: float = 1.0,
) -> float:
    """
    Compute Precision at K.
    
    Precision@K = (# relevant in top K) / K
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted scores
        k: Cutoff position
        threshold: Relevance threshold (items with score >= threshold are relevant)
        
    Returns:
        Precision@K score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Sort by predicted scores
    order = np.argsort(-y_pred)[:k]
    
    # Count relevant items
    relevant = np.sum(y_true[order] >= threshold)
    
    return float(relevant / k)


def recall_at_k(
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
    k: int = 10,
    threshold: float = 1.0,
) -> float:
    """
    Compute Recall at K.
    
    Recall@K = (# relevant in top K) / (total # relevant)
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted scores
        k: Cutoff position
        threshold: Relevance threshold
        
    Returns:
        Recall@K score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Total relevant items
    total_relevant = np.sum(y_true >= threshold)
    
    if total_relevant == 0:
        return 0.0
    
    # Sort by predicted scores
    order = np.argsort(-y_pred)[:k]
    
    # Count relevant in top K
    relevant_in_k = np.sum(y_true[order] >= threshold)
    
    return float(relevant_in_k / total_relevant)


def average_precision(
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
    threshold: float = 1.0,
) -> float:
    """
    Compute Average Precision.
    
    AP = Σ (P@k * rel_k) / (# relevant items)
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted scores
        threshold: Relevance threshold
        
    Returns:
        AP score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Sort by predicted scores
    order = np.argsort(-y_pred)
    y_true_sorted = y_true[order]
    
    # Binary relevance
    relevant = (y_true_sorted >= threshold).astype(float)
    
    total_relevant = relevant.sum()
    if total_relevant == 0:
        return 0.0
    
    # Compute precision at each relevant position
    precision_sum = 0.0
    relevant_count = 0
    
    for i, rel in enumerate(relevant):
        if rel > 0:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    
    return float(precision_sum / total_relevant)


def map_score(
    y_true_list: list[np.ndarray],
    y_pred_list: list[np.ndarray],
    threshold: float = 1.0,
) -> float:
    """
    Compute Mean Average Precision across queries.
    
    Args:
        y_true_list: List of true relevance arrays (one per query)
        y_pred_list: List of predicted score arrays (one per query)
        threshold: Relevance threshold
        
    Returns:
        MAP score
    """
    ap_scores = [
        average_precision(y_true, y_pred, threshold)
        for y_true, y_pred in zip(y_true_list, y_pred_list)
    ]
    
    return float(np.mean(ap_scores))


def reciprocal_rank(
    y_true: np.ndarray | Sequence[float],
    y_pred: np.ndarray | Sequence[float],
    threshold: float = 1.0,
) -> float:
    """
    Compute Reciprocal Rank.
    
    RR = 1 / (rank of first relevant item)
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted scores
        threshold: Relevance threshold
        
    Returns:
        RR score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Sort by predicted scores
    order = np.argsort(-y_pred)
    y_true_sorted = y_true[order]
    
    # Find first relevant item
    relevant_positions = np.where(y_true_sorted >= threshold)[0]
    
    if len(relevant_positions) == 0:
        return 0.0
    
    first_relevant = relevant_positions[0]
    return 1.0 / (first_relevant + 1)


def mrr_score(
    y_true_list: list[np.ndarray],
    y_pred_list: list[np.ndarray],
    threshold: float = 1.0,
) -> float:
    """
    Compute Mean Reciprocal Rank across queries.
    
    Args:
        y_true_list: List of true relevance arrays
        y_pred_list: List of predicted score arrays
        threshold: Relevance threshold
        
    Returns:
        MRR score
    """
    rr_scores = [
        reciprocal_rank(y_true, y_pred, threshold)
        for y_true, y_pred in zip(y_true_list, y_pred_list)
    ]
    
    return float(np.mean(rr_scores))


@dataclass
class RankingMetrics:
    """
    Comprehensive ranking evaluation.
    
    Computes all metrics at various cutoffs and provides aggregated results.
    """
    
    cutoffs: list[int] = field(default_factory=lambda: [5, 10, 20])
    threshold: float = 1.0
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Evaluate ranking performance.
        
        Args:
            y_true: True relevance scores
            y_pred: Predicted scores
            groups: Query group sizes (optional)
            
        Returns:
            Dictionary of metric names to values
        """
        if groups is None:
            # Single query
            return self._evaluate_single(y_true, y_pred)
        
        # Multiple queries
        return self._evaluate_multiple(y_true, y_pred, groups)
    
    def _evaluate_single(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate metrics for a single query."""
        metrics = {}
        
        for k in self.cutoffs:
            metrics[f"ndcg@{k}"] = ndcg_at_k(y_true, y_pred, k)
            metrics[f"precision@{k}"] = precision_at_k(y_true, y_pred, k, self.threshold)
            metrics[f"recall@{k}"] = recall_at_k(y_true, y_pred, k, self.threshold)
        
        metrics["map"] = average_precision(y_true, y_pred, self.threshold)
        metrics["mrr"] = reciprocal_rank(y_true, y_pred, self.threshold)
        
        return metrics
    
    def _evaluate_multiple(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate metrics across multiple queries."""
        # Split by groups
        y_true_list = []
        y_pred_list = []
        
        offset = 0
        for group_size in groups:
            if group_size > 0:
                y_true_list.append(y_true[offset:offset + group_size])
                y_pred_list.append(y_pred[offset:offset + group_size])
            offset += group_size
        
        # Compute per-query metrics and average
        all_metrics: dict[str, list[float]] = {}
        
        for y_t, y_p in zip(y_true_list, y_pred_list):
            single_metrics = self._evaluate_single(y_t, y_p)
            for name, value in single_metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = []
                all_metrics[name].append(value)
        
        # Average
        return {name: float(np.mean(values)) for name, values in all_metrics.items()}
    
    def evaluate_with_confidence(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> dict[str, tuple[float, float, float]]:
        """
        Evaluate with bootstrap confidence intervals.
        
        Args:
            y_true: True relevance scores
            y_pred: Predicted scores
            groups: Query group sizes
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Dict of metric -> (mean, lower, upper)
        """
        # Base evaluation
        base_metrics = self._evaluate_multiple(y_true, y_pred, groups)
        
        # Bootstrap
        n_queries = len(groups)
        bootstrap_results: dict[str, list[float]] = {k: [] for k in base_metrics}
        
        # Split data by queries
        y_true_split = []
        y_pred_split = []
        offset = 0
        for group_size in groups:
            y_true_split.append(y_true[offset:offset + group_size])
            y_pred_split.append(y_pred[offset:offset + group_size])
            offset += group_size
        
        for _ in range(n_bootstrap):
            # Sample queries with replacement
            indices = np.random.choice(n_queries, size=n_queries, replace=True)
            
            sampled_true = [y_true_split[i] for i in indices]
            sampled_pred = [y_pred_split[i] for i in indices]
            sampled_groups = np.array([len(s) for s in sampled_true])
            
            # Concatenate
            y_true_boot = np.concatenate(sampled_true)
            y_pred_boot = np.concatenate(sampled_pred)
            
            # Evaluate
            boot_metrics = self._evaluate_multiple(y_true_boot, y_pred_boot, sampled_groups)
            
            for name, value in boot_metrics.items():
                bootstrap_results[name].append(value)
        
        # Compute confidence intervals
        alpha = 1 - confidence
        results = {}
        
        for name, values in bootstrap_results.items():
            values_arr = np.array(values)
            lower = np.percentile(values_arr, 100 * alpha / 2)
            upper = np.percentile(values_arr, 100 * (1 - alpha / 2))
            results[name] = (base_metrics[name], lower, upper)
        
        return results


def compute_lift(
    baseline_metric: float,
    treatment_metric: float,
) -> float:
    """
    Compute relative lift of treatment over baseline.
    
    Lift = (treatment - baseline) / baseline * 100
    
    Returns:
        Percentage lift
    """
    if baseline_metric == 0:
        return 0.0
    
    return (treatment_metric - baseline_metric) / baseline_metric * 100
