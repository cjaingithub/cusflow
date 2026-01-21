"""
Feature engineering for ranking models.

Combines item features, user features, context features, and GenAI features
into a unified feature vector for the ranking model.
"""

from typing import Any

import numpy as np

from src.config import Domain, DomainConfig, get_settings
from src.data.schemas import ContextFeatures, Item, ItemFeatures, User, UserFeatures


class FeatureEngineer:
    """
    Transform raw features into model-ready feature vectors.
    
    Features include:
    1. Item features (numeric and categorical)
    2. User features
    3. Context features
    4. Cross features (user-item interactions)
    5. GenAI features (embedding similarities)
    """
    
    def __init__(self, domain: Domain | None = None):
        self.settings = get_settings()
        self.domain = domain or self.settings.domain
        self.domain_config = DomainConfig.get(self.domain)
        
        # Feature configuration
        self.item_feature_names = self.domain_config["features"]
        self.user_feature_names = self.domain_config["user_features"]
        self.context_feature_names = self.domain_config["context_features"]
        
        # Encoders for categorical features (populated during fit)
        self._categorical_encoders: dict[str, dict[str, int]] = {}
        self._feature_means: dict[str, float] = {}
        self._feature_stds: dict[str, float] = {}
        
        self._all_feature_names: list[str] = []
        self._is_fitted = False
    
    def fit(
        self,
        items: list[Item],
        users: list[User] | None = None,
    ) -> "FeatureEngineer":
        """
        Fit feature transformers on training data.
        
        Args:
            items: Training items
            users: Training users (optional)
        """
        # Collect all feature values for statistics
        item_features: dict[str, list[Any]] = {name: [] for name in self.item_feature_names}
        
        for item in items:
            for name in self.item_feature_names:
                value = item.features.features.get(name)
                if value is not None:
                    item_features[name].append(value)
        
        # Compute statistics and encoders
        for name, values in item_features.items():
            if not values:
                continue
            
            if isinstance(values[0], (int, float)):
                # Numeric feature - compute mean/std for normalization
                arr = np.array(values, dtype=float)
                self._feature_means[name] = float(np.nanmean(arr))
                self._feature_stds[name] = float(np.nanstd(arr)) or 1.0
            else:
                # Categorical feature - create encoder
                unique_values = list(set(str(v) for v in values))
                self._categorical_encoders[name] = {v: i for i, v in enumerate(unique_values)}
        
        # Build feature name list
        self._build_feature_names()
        self._is_fitted = True
        
        return self
    
    def _build_feature_names(self) -> None:
        """Build the full list of feature names."""
        names = []
        
        # Item features
        for name in self.item_feature_names:
            if name in self._categorical_encoders:
                # One-hot encoded
                for val in self._categorical_encoders[name]:
                    names.append(f"item_{name}_{val}")
            else:
                names.append(f"item_{name}")
        
        # GenAI features
        names.extend([
            "genai_quality_score",
            "genai_embedding_sim",
            "genai_summary_length",
        ])
        
        # User-item cross features
        names.extend([
            "cross_user_item_history",
            "cross_price_vs_avg_spend",
            "cross_rating_vs_preferred",
        ])
        
        # Context features
        names.extend([
            "ctx_is_mobile",
            "ctx_is_weekend",
            "ctx_hour_of_day",
        ])
        
        self._all_feature_names = names
    
    @property
    def feature_names(self) -> list[str]:
        """Get all feature names."""
        return self._all_feature_names
    
    @property
    def num_features(self) -> int:
        """Get number of features."""
        return len(self._all_feature_names)
    
    def transform(
        self,
        items: list[Item],
        user: User | None = None,
        context: ContextFeatures | None = None,
        user_embedding: np.ndarray | None = None,
        item_embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Transform items into feature matrix.
        
        Args:
            items: Items to transform
            user: User context (optional)
            context: Request context (optional)
            user_embedding: User preference embedding for similarity
            item_embeddings: Item embeddings for similarity
            
        Returns:
            Feature matrix (n_items, n_features)
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer not fitted. Call fit() first.")
        
        features = []
        
        for i, item in enumerate(items):
            item_features = self._transform_item(item)
            genai_features = self._transform_genai(
                item, 
                user_embedding,
                item_embeddings[i] if item_embeddings is not None else None,
            )
            cross_features = self._transform_cross(item, user)
            context_features = self._transform_context(context)
            
            features.append(
                item_features + genai_features + cross_features + context_features
            )
        
        return np.array(features, dtype=np.float32)
    
    def _transform_item(self, item: Item) -> list[float]:
        """Transform item features."""
        features = []
        
        for name in self.item_feature_names:
            value = item.features.features.get(name)
            
            if name in self._categorical_encoders:
                # One-hot encode
                encoder = self._categorical_encoders[name]
                for cat_val in encoder:
                    features.append(1.0 if str(value) == cat_val else 0.0)
            else:
                # Numeric - normalize
                if value is None:
                    features.append(0.0)
                else:
                    mean = self._feature_means.get(name, 0.0)
                    std = self._feature_stds.get(name, 1.0)
                    features.append((float(value) - mean) / std)
        
        return features
    
    def _transform_genai(
        self,
        item: Item,
        user_embedding: np.ndarray | None,
        item_embedding: np.ndarray | None,
    ) -> list[float]:
        """Transform GenAI-related features."""
        features = []
        
        # Quality score from features
        features.append(item.features.quality_score)
        
        # Embedding similarity (if embeddings available)
        if user_embedding is not None and item_embedding is not None:
            sim = np.dot(user_embedding, item_embedding) / (
                np.linalg.norm(user_embedding) * np.linalg.norm(item_embedding) + 1e-8
            )
            features.append(float(sim))
        else:
            features.append(0.0)
        
        # Summary length (proxy for content richness)
        summary = item.features.summary or ""
        features.append(len(summary) / 500)  # Normalize by expected max
        
        return features
    
    def _transform_cross(self, item: Item, user: User | None) -> list[float]:
        """Transform user-item cross features."""
        if user is None:
            return [0.0, 0.0, 0.0]
        
        features = []
        user_feats = user.features.features
        item_feats = item.features.features
        
        # User-item interaction history (placeholder)
        features.append(0.0)
        
        # Price vs user's average spend
        if "avg_spend" in user_feats and "price_per_night" in item_feats:
            avg_spend = float(user_feats["avg_spend"])
            price = float(item_feats["price_per_night"])
            features.append((price - avg_spend) / (avg_spend + 1))
        elif "avg_order_value" in user_feats and "price" in item_feats:
            avg = float(user_feats["avg_order_value"])
            price = float(item_feats["price"])
            features.append((price - avg) / (avg + 1))
        else:
            features.append(0.0)
        
        # Rating preference match
        if "preferred_star_rating" in user_feats and "star_rating" in item_feats:
            pref = float(user_feats["preferred_star_rating"])
            actual = float(item_feats["star_rating"])
            features.append(1.0 - abs(pref - actual) / 5)
        elif "rating" in item_feats:
            features.append(float(item_feats["rating"]) / 5)
        else:
            features.append(0.0)
        
        return features
    
    def _transform_context(self, context: ContextFeatures | None) -> list[float]:
        """Transform context features."""
        if context is None:
            return [0.0, 0.0, 0.0]
        
        features = []
        
        # Is mobile
        features.append(1.0 if context.device_type == "mobile" else 0.0)
        
        # Is weekend
        is_weekend = context.timestamp.weekday() >= 5
        features.append(1.0 if is_weekend else 0.0)
        
        # Hour of day (normalized)
        features.append(context.timestamp.hour / 24)
        
        return features
    
    def get_feature_vector(
        self,
        item: Item,
        user: User | None = None,
        context: ContextFeatures | None = None,
    ) -> np.ndarray:
        """Get feature vector for a single item."""
        return self.transform([item], user, context)[0]
