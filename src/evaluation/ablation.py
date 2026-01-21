"""
Ablation Study for Feature Importance Analysis.

Compare model performance with and without specific features
to understand their contribution to ranking quality.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm

from src.evaluation.metrics import RankingMetrics, compute_lift


@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    
    # Feature groups to ablate
    feature_groups: dict[str, list[str]] = field(default_factory=dict)
    
    # Metrics to evaluate
    metrics: list[str] = field(default_factory=lambda: ["ndcg@10", "map", "recall@10"])
    
    # Number of cross-validation folds
    n_folds: int = 5


@dataclass
class AblationResult:
    """Results from ablation study."""
    
    # Full model metrics
    full_model_metrics: dict[str, float]
    
    # Ablated model metrics (feature_group -> metrics)
    ablated_metrics: dict[str, dict[str, float]]
    
    # Feature importance (contribution to each metric)
    feature_importance: dict[str, dict[str, float]]
    
    # Relative drops when removing features
    relative_drops: dict[str, dict[str, float]]


class AblationStudy:
    """
    Conduct ablation study to analyze feature contributions.
    
    Features:
    - Remove feature groups one at a time
    - Compare against full model
    - Compute importance scores
    """
    
    def __init__(
        self,
        model_class: Any,
        model_params: dict[str, Any] | None = None,
        config: AblationConfig | None = None,
    ):
        self.model_class = model_class
        self.model_params = model_params or {}
        self.config = config or AblationConfig()
        
        self.metrics_calculator = RankingMetrics()
    
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        feature_names: list[str],
        show_progress: bool = True,
    ) -> AblationResult:
        """
        Run ablation study.
        
        Args:
            X: Feature matrix
            y: Labels
            groups: Query group sizes
            feature_names: Names of features
            show_progress: Show progress bar
            
        Returns:
            AblationResult with importance scores
        """
        # Define feature groups if not specified
        if not self.config.feature_groups:
            self.config.feature_groups = self._auto_group_features(feature_names)
        
        # Train and evaluate full model
        full_metrics = self._train_and_evaluate(X, y, groups, feature_names)
        
        # Ablate each feature group
        ablated_metrics = {}
        feature_groups = list(self.config.feature_groups.items())
        
        iterator = tqdm(feature_groups, desc="Ablating") if show_progress else feature_groups
        
        for group_name, group_features in iterator:
            # Get indices of features to remove
            remove_indices = [
                i for i, name in enumerate(feature_names)
                if any(gf in name for gf in group_features)
            ]
            
            if not remove_indices:
                continue
            
            # Create ablated feature matrix
            keep_indices = [i for i in range(X.shape[1]) if i not in remove_indices]
            X_ablated = X[:, keep_indices]
            ablated_names = [feature_names[i] for i in keep_indices]
            
            # Train and evaluate
            metrics = self._train_and_evaluate(X_ablated, y, groups, ablated_names)
            ablated_metrics[group_name] = metrics
        
        # Compute importance and relative drops
        feature_importance = {}
        relative_drops = {}
        
        for group_name, metrics in ablated_metrics.items():
            feature_importance[group_name] = {}
            relative_drops[group_name] = {}
            
            for metric_name in full_metrics:
                full_value = full_metrics[metric_name]
                ablated_value = metrics[metric_name]
                
                # Importance = drop in performance when removed
                drop = full_value - ablated_value
                feature_importance[group_name][metric_name] = max(0, drop)
                
                # Relative drop
                if full_value > 0:
                    relative_drops[group_name][metric_name] = (drop / full_value) * 100
                else:
                    relative_drops[group_name][metric_name] = 0
        
        return AblationResult(
            full_model_metrics=full_metrics,
            ablated_metrics=ablated_metrics,
            feature_importance=feature_importance,
            relative_drops=relative_drops,
        )
    
    def _train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Train model and evaluate on held-out data."""
        from sklearn.model_selection import GroupKFold
        
        # Convert groups to query IDs
        query_ids = np.repeat(np.arange(len(groups)), groups)
        
        cv = GroupKFold(n_splits=min(self.config.n_folds, len(groups)))
        
        all_metrics: dict[str, list[float]] = {}
        
        for train_idx, val_idx in cv.split(X, y, query_ids):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Compute groups for train/val
            train_qids = query_ids[train_idx]
            val_qids = query_ids[val_idx]
            
            train_groups = np.bincount(train_qids)
            train_groups = train_groups[train_groups > 0]
            
            val_groups = np.bincount(val_qids - val_qids.min())
            val_groups = val_groups[val_groups > 0]
            
            # Train model
            model = self.model_class(**self.model_params)
            model.fit(X_train, y_train, train_groups, feature_names=feature_names)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Evaluate
            metrics = self.metrics_calculator.evaluate(y_val, y_pred, val_groups)
            
            for name, value in metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = []
                all_metrics[name].append(value)
        
        # Average across folds
        return {name: float(np.mean(values)) for name, values in all_metrics.items()}
    
    def _auto_group_features(self, feature_names: list[str]) -> dict[str, list[str]]:
        """Automatically group features by prefix."""
        groups: dict[str, list[str]] = {}
        
        for name in feature_names:
            # Extract prefix (e.g., "item_", "user_", "genai_")
            parts = name.split("_")
            if len(parts) > 1:
                prefix = parts[0]
            else:
                prefix = "other"
            
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(name)
        
        return groups
    
    def generate_report(self, result: AblationResult) -> str:
        """Generate human-readable ablation report."""
        lines = [
            "=" * 70,
            "Ablation Study Report",
            "=" * 70,
            "",
            "Full Model Performance:",
            "-" * 40,
        ]
        
        for metric, value in result.full_model_metrics.items():
            lines.append(f"  {metric:20s}: {value:.4f}")
        
        lines.extend([
            "",
            "Feature Group Importance (drop when removed):",
            "-" * 70,
        ])
        
        # Sort by importance for main metric
        main_metric = "ndcg@10" if "ndcg@10" in result.full_model_metrics else list(result.full_model_metrics.keys())[0]
        
        sorted_groups = sorted(
            result.feature_importance.items(),
            key=lambda x: x[1].get(main_metric, 0),
            reverse=True,
        )
        
        for group_name, importance in sorted_groups:
            lines.append(f"\n  {group_name}:")
            drops = result.relative_drops[group_name]
            
            for metric in importance:
                imp = importance[metric]
                drop_pct = drops[metric]
                lines.append(f"    {metric:15s}: -{imp:.4f} ({drop_pct:+.1f}%)")
        
        lines.extend([
            "",
            "=" * 70,
            "",
            "Key Findings:",
            "-" * 40,
        ])
        
        # Find most important feature group
        if sorted_groups:
            top_group = sorted_groups[0][0]
            top_drop = result.relative_drops[top_group].get(main_metric, 0)
            lines.append(
                f"  Most important feature group: '{top_group}' "
                f"({top_drop:.1f}% drop in {main_metric} when removed)"
            )
        
        # Check GenAI contribution
        genai_drop = result.relative_drops.get("genai", {}).get(main_metric, 0)
        if genai_drop > 0:
            lines.append(
                f"  GenAI features contribution: {genai_drop:.1f}% improvement in {main_metric}"
            )
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


def compare_with_without_genai(
    X_with: np.ndarray,
    X_without: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_class: Any,
    model_params: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Quick comparison of model with and without GenAI features.
    
    Args:
        X_with: Features including GenAI
        X_without: Features without GenAI
        y: Labels
        groups: Query groups
        model_class: Model class to use
        model_params: Model parameters
        
    Returns:
        Uplift metrics
    """
    metrics_calculator = RankingMetrics()
    model_params = model_params or {}
    
    # Train with GenAI
    model_with = model_class(**model_params)
    model_with.fit(X_with, y, groups)
    pred_with = model_with.predict(X_with)
    metrics_with = metrics_calculator.evaluate(y, pred_with, groups)
    
    # Train without GenAI
    model_without = model_class(**model_params)
    model_without.fit(X_without, y, groups)
    pred_without = model_without.predict(X_without)
    metrics_without = metrics_calculator.evaluate(y, pred_without, groups)
    
    # Compute uplift
    uplift = {}
    for metric in metrics_with:
        uplift[f"{metric}_with_genai"] = metrics_with[metric]
        uplift[f"{metric}_without_genai"] = metrics_without[metric]
        uplift[f"{metric}_uplift"] = compute_lift(
            metrics_without[metric],
            metrics_with[metric],
        )
    
    return uplift
