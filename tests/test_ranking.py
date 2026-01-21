"""
Tests for the ranking module.
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    average_precision,
    mrr_score,
    RankingMetrics,
)


class TestMetrics:
    """Test ranking metrics."""
    
    def test_dcg_at_k_basic(self):
        """Test DCG calculation."""
        relevance = np.array([3, 2, 1, 0])
        dcg = dcg_at_k(relevance, k=4)
        
        # DCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4) + (2^0-1)/log2(5)
        expected = 7/1 + 3/1.585 + 1/2 + 0
        assert abs(dcg - expected) < 0.01
    
    def test_dcg_at_k_empty(self):
        """Test DCG with empty input."""
        assert dcg_at_k(np.array([]), k=5) == 0.0
    
    def test_ndcg_at_k_perfect(self):
        """Test NDCG with perfect ranking."""
        y_true = np.array([3, 2, 1, 0])
        y_pred = np.array([4, 3, 2, 1])  # Same ranking as true
        
        ndcg = ndcg_at_k(y_true, y_pred, k=4)
        assert abs(ndcg - 1.0) < 0.001
    
    def test_ndcg_at_k_worst(self):
        """Test NDCG with reversed ranking."""
        y_true = np.array([3, 2, 1, 0])
        y_pred = np.array([1, 2, 3, 4])  # Reversed ranking
        
        ndcg = ndcg_at_k(y_true, y_pred, k=4)
        assert ndcg < 1.0
    
    def test_ndcg_at_k_no_relevant(self):
        """Test NDCG when no relevant items."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([4, 3, 2, 1])
        
        ndcg = ndcg_at_k(y_true, y_pred, k=4)
        assert ndcg == 0.0
    
    def test_precision_at_k(self):
        """Test precision@k."""
        y_true = np.array([1, 0, 1, 0, 1])  # 3 relevant
        y_pred = np.array([5, 4, 3, 2, 1])  # First two have rel=1, 0
        
        # Top 2: items with pred 5, 4 -> rel 1, 0 -> precision = 1/2
        prec = precision_at_k(y_true, y_pred, k=2, threshold=1.0)
        assert abs(prec - 0.5) < 0.001
    
    def test_recall_at_k(self):
        """Test recall@k."""
        y_true = np.array([1, 0, 1, 0, 1])  # 3 relevant
        y_pred = np.array([5, 4, 3, 2, 1])  # Top 3 has 2 relevant
        
        # Top 3 contains 2 relevant out of 3 total -> recall = 2/3
        rec = recall_at_k(y_true, y_pred, k=3, threshold=1.0)
        assert abs(rec - 2/3) < 0.001
    
    def test_average_precision(self):
        """Test average precision."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([5, 4, 3, 2, 1])
        
        ap = average_precision(y_true, y_pred, threshold=1.0)
        # P@1 * rel_1 + P@2 * rel_2 + ... / num_rel
        # = (1/1 * 1 + 0 + 2/3 * 1 + 0 + 3/5 * 1) / 3
        expected = (1 + 2/3 + 3/5) / 3
        assert abs(ap - expected) < 0.001


class TestRankingMetrics:
    """Test the RankingMetrics class."""
    
    def test_evaluate_single_query(self):
        """Test evaluation on single query."""
        y_true = np.array([3, 2, 1, 0, 0])
        y_pred = np.array([5, 4, 3, 2, 1])
        
        metrics = RankingMetrics(cutoffs=[5])
        results = metrics.evaluate(y_true, y_pred)
        
        assert "ndcg@5" in results
        assert "precision@5" in results
        assert "recall@5" in results
        assert "map" in results
        assert "mrr" in results
    
    def test_evaluate_multiple_queries(self):
        """Test evaluation on multiple queries."""
        # Two queries
        y_true = np.array([3, 2, 1, 2, 1, 0])  # Query 1: [3,2,1], Query 2: [2,1,0]
        y_pred = np.array([3, 2, 1, 3, 2, 1])
        groups = np.array([3, 3])
        
        metrics = RankingMetrics(cutoffs=[3])
        results = metrics.evaluate(y_true, y_pred, groups)
        
        assert "ndcg@3" in results
        assert all(0 <= v <= 1 for v in results.values() if "ndcg" in v or "precision" in v)


class TestBiasCorrection:
    """Test bias correction components."""
    
    def test_position_bias_corrector(self):
        """Test position bias estimation."""
        from src.ranking.bias_correction import PositionBiasCorrector
        
        # Simulate position bias data
        np.random.seed(42)
        positions = np.random.randint(1, 11, size=1000)
        
        # Click probability decreases with position
        click_probs = 1 / (positions ** 0.5)
        clicks = np.random.binomial(1, click_probs)
        
        corrector = PositionBiasCorrector()
        corrector.fit(positions, clicks)
        
        # Higher positions should have higher examination probability
        assert corrector.get_propensity(1) > corrector.get_propensity(10)
    
    def test_ipw_weights(self):
        """Test inverse propensity weighting."""
        from src.ranking.bias_correction import InversePropensityWeighting
        
        positions = np.array([1, 2, 3, 4, 5])
        clicks = np.array([1, 1, 0, 0, 0])
        
        ipw = InversePropensityWeighting()
        ipw.fit(positions, clicks)
        weights = ipw.compute_weights(positions, clicks)
        
        # Weights should be higher for clicked items at lower positions
        assert weights[0] < weights[1]  # Position 1 click vs position 2 click
        assert len(weights) == len(positions)


class TestLambdaMART:
    """Test LambdaMART ranker."""
    
    def test_fit_predict(self):
        """Test basic fit and predict."""
        from src.ranking.lambdamart import LambdaMARTRanker
        
        # Create synthetic data
        np.random.seed(42)
        n_queries = 20
        items_per_query = 10
        n_features = 5
        
        X = np.random.randn(n_queries * items_per_query, n_features)
        y = np.random.randint(0, 5, size=n_queries * items_per_query)
        groups = np.array([items_per_query] * n_queries)
        
        # Train model
        model = LambdaMARTRanker(num_boost_round=10)
        model.fit(X, y, groups)
        
        # Predict
        scores = model.predict(X)
        
        assert len(scores) == len(y)
        assert model.model is not None
    
    def test_rank(self):
        """Test ranking items."""
        from src.ranking.lambdamart import LambdaMARTRanker
        
        np.random.seed(42)
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 5, size=20)
        groups = np.array([20])
        
        model = LambdaMARTRanker(num_boost_round=10)
        model.fit(X, y, groups)
        
        item_ids = [f"item_{i}" for i in range(5)]
        ranked = model.rank(X[:5], item_ids, top_k=3)
        
        assert len(ranked) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in ranked)


class TestFeatureEngineering:
    """Test feature engineering."""
    
    def test_feature_engineer(self):
        """Test feature transformation."""
        from src.data.schemas import Item, ItemFeatures
        from src.ranking.feature_engineering import FeatureEngineer
        
        # Create test items
        items = [
            Item(
                item_id="1",
                domain="hotel",
                name="Test Hotel",
                features=ItemFeatures(
                    features={"star_rating": 4, "review_score": 4.5, "price_per_night": 150},
                    quality_score=0.8,
                ),
            ),
            Item(
                item_id="2",
                domain="hotel",
                name="Another Hotel",
                features=ItemFeatures(
                    features={"star_rating": 3, "review_score": 4.0, "price_per_night": 100},
                    quality_score=0.7,
                ),
            ),
        ]
        
        fe = FeatureEngineer()
        fe.fit(items)
        
        X = fe.transform(items)
        
        assert X.shape[0] == 2
        assert X.shape[1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
