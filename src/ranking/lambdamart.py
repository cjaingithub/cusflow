"""
LambdaMART implementation for Learning to Rank.

Uses LightGBM's implementation of LambdaMART for efficient training
with support for position bias correction via sample weighting.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

from src.config import get_settings


class LambdaMARTRanker:
    """
    LambdaMART ranker using LightGBM.
    
    Features:
    - Listwise loss optimization (LambdaRank)
    - Support for inverse propensity weighting
    - Feature importance analysis
    - Cross-validation with query groups
    """
    
    DEFAULT_PARAMS = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10, 20],
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 20,
        "max_depth": 8,
        "verbose": -1,
        "num_threads": -1,
        "seed": 42,
    }
    
    def __init__(
        self,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
    ):
        self.settings = get_settings()
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        
        self.model: lgb.Booster | None = None
        self.feature_names: list[str] = []
        self.feature_importance_: dict[str, float] = {}
        self._best_iteration: int = 0
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        sample_weight: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        groups_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "LambdaMARTRanker":
        """
        Train LambdaMART model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Relevance labels (n_samples,)
            groups: Query group sizes (n_queries,) - sum must equal n_samples
            sample_weight: Optional weights for bias correction
            X_val: Validation features
            y_val: Validation labels
            groups_val: Validation group sizes
            feature_names: Names of features for importance analysis
            
        Returns:
            self
        """
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X, 
            label=y, 
            group=groups,
            weight=sample_weight,
            feature_name=self.feature_names,
        )
        
        valid_sets = [train_data]
        valid_names = ["train"]
        
        if X_val is not None and y_val is not None and groups_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                group=groups_val,
                feature_name=self.feature_names,
                reference=train_data,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds),
                lgb.log_evaluation(50),
            ],
        )
        
        self._best_iteration = self.model.best_iteration
        
        # Compute feature importance
        importance = self.model.feature_importance(importance_type="gain")
        total = importance.sum()
        self.feature_importance_ = {
            name: float(imp / total) if total > 0 else 0.0
            for name, imp in zip(self.feature_names, importance)
        }
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ranking scores.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Ranking scores (n_samples,)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return self.model.predict(X, num_iteration=self._best_iteration)
    
    def rank(
        self, 
        X: np.ndarray, 
        item_ids: list[str],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Rank items by predicted score.
        
        Args:
            X: Features for each item
            item_ids: Item identifiers
            top_k: Number of top items to return
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        scores = self.predict(X)
        
        # Sort by score descending
        sorted_indices = np.argsort(-scores)
        
        results = [(item_ids[i], float(scores[i])) for i in sorted_indices]
        
        if top_k:
            results = results[:top_k]
        
        return results
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: int = 5,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """
        Perform cross-validation with query-aware splits.
        
        Args:
            X: Features
            y: Labels
            groups: Query group sizes
            n_splits: Number of CV folds
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary of metric lists across folds
        """
        # Convert groups to query IDs for GroupKFold
        query_ids = np.repeat(np.arange(len(groups)), groups)
        
        cv = GroupKFold(n_splits=n_splits)
        metrics: dict[str, list[float]] = {"ndcg@10": [], "ndcg@5": []}
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, query_ids)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Compute groups for train/val
            train_qids = query_ids[train_idx]
            val_qids = query_ids[val_idx]
            
            train_groups = np.bincount(train_qids[train_qids >= 0])
            train_groups = train_groups[train_groups > 0]
            
            val_groups = np.bincount(val_qids - val_qids.min())
            val_groups = val_groups[val_groups > 0]
            
            # Get sample weights for training set
            sw_train = sample_weight[train_idx] if sample_weight is not None else None
            
            # Train model
            self.fit(
                X_train, y_train, train_groups,
                sample_weight=sw_train,
                X_val=X_val, y_val=y_val, groups_val=val_groups,
            )
            
            # Evaluate on validation set
            val_scores = self.predict(X_val)
            
            # Compute NDCG for each query
            ndcg_scores = self._compute_ndcg_per_query(y_val, val_scores, val_groups)
            metrics["ndcg@10"].append(np.mean(ndcg_scores.get(10, [0])))
            metrics["ndcg@5"].append(np.mean(ndcg_scores.get(5, [0])))
            
            print(f"Fold {fold + 1}: NDCG@10 = {metrics['ndcg@10'][-1]:.4f}")
        
        return metrics
    
    def _compute_ndcg_per_query(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        ks: list[int] = [5, 10],
    ) -> dict[int, list[float]]:
        """Compute NDCG for each query at various cutoffs."""
        from src.evaluation.metrics import ndcg_at_k
        
        results = {k: [] for k in ks}
        offset = 0
        
        for group_size in groups:
            if group_size == 0:
                continue
                
            y_true_q = y_true[offset:offset + group_size]
            y_pred_q = y_pred[offset:offset + group_size]
            
            for k in ks:
                ndcg = ndcg_at_k(y_true_q, y_pred_q, k)
                results[k].append(ndcg)
            
            offset += group_size
        
        return results
    
    def get_feature_importance(
        self, 
        top_k: int | None = None,
    ) -> dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            top_k: Return only top K features
            
        Returns:
            Feature name -> importance mapping
        """
        sorted_importance = dict(
            sorted(self.feature_importance_.items(), key=lambda x: -x[1])
        )
        
        if top_k:
            return dict(list(sorted_importance.items())[:top_k])
        
        return sorted_importance
    
    def save(self, path: Path | None = None) -> None:
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        path = path or self.settings.ranking_model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        self.model.save_model(str(path.with_suffix(".lgb")))
        
        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance_,
            "best_iteration": self._best_iteration,
            "params": self.params,
        }
        
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Also save as joblib for full object serialization
        joblib.dump(self, path)
    
    def load(self, path: Path | None = None) -> "LambdaMARTRanker":
        """Load model from disk."""
        path = path or self.settings.ranking_model_path
        
        # Try loading full joblib first
        if path.exists():
            loaded = joblib.load(path)
            self.__dict__.update(loaded.__dict__)
            return self
        
        # Fallback to separate files
        lgb_path = path.with_suffix(".lgb")
        if lgb_path.exists():
            self.model = lgb.Booster(model_file=str(lgb_path))
            
            meta_path = path.with_suffix(".json")
            if meta_path.exists():
                with open(meta_path) as f:
                    metadata = json.load(f)
                
                self.feature_names = metadata["feature_names"]
                self.feature_importance_ = metadata["feature_importance"]
                self._best_iteration = metadata["best_iteration"]
                self.params = metadata.get("params", self.params)
        
        return self


class XGBoostRanker:
    """
    Alternative ranker using XGBoost.
    
    Useful for comparison or when LightGBM is not available.
    """
    
    DEFAULT_PARAMS = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "eta": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    
    def __init__(
        self,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
    ):
        import xgboost as xgb
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model: xgb.Booster | None = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        groups_val: np.ndarray | None = None,
    ) -> "XGBoostRanker":
        """Train XGBoost ranker."""
        import xgboost as xgb
        
        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(groups)
        
        evals = [(dtrain, "train")]
        
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            dval.set_group(groups_val)
            evals.append((dval, "valid"))
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=50,
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ranking scores."""
        import xgboost as xgb
        
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
