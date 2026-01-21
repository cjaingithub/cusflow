"""
Position bias correction for Learning to Rank.

Implements methods to correct for the fact that items shown at higher
positions receive more clicks regardless of relevance.
"""

from typing import Literal

import numpy as np
from scipy.optimize import minimize


class PositionBiasCorrector:
    """
    Estimate and correct for position bias in click data.
    
    Uses the cascade model assumption: users examine positions sequentially
    and have a position-dependent probability of examining each position.
    """
    
    def __init__(
        self,
        max_position: int = 20,
        bias_model: Literal["power", "exponential", "learned"] = "power",
    ):
        self.max_position = max_position
        self.bias_model = bias_model
        
        # Position examination probabilities
        self.examination_probs: np.ndarray | None = None
        
        # Power law parameter
        self.power_param: float = 1.0
    
    def fit(
        self,
        positions: np.ndarray,
        clicks: np.ndarray,
        relevance_estimates: np.ndarray | None = None,
    ) -> "PositionBiasCorrector":
        """
        Estimate position bias from click data.
        
        Args:
            positions: Position of each impression (1-indexed)
            clicks: Binary click indicator
            relevance_estimates: Optional prior relevance estimates
            
        Returns:
            self
        """
        if self.bias_model == "power":
            self._fit_power_law(positions, clicks)
        elif self.bias_model == "exponential":
            self._fit_exponential(positions, clicks)
        elif self.bias_model == "learned":
            self._fit_em(positions, clicks, relevance_estimates)
        
        return self
    
    def _fit_power_law(self, positions: np.ndarray, clicks: np.ndarray) -> None:
        """Fit power law bias: P(examine|position) = 1/position^alpha."""
        # Grid search for alpha
        best_alpha = 1.0
        best_ll = float("-inf")
        
        for alpha in np.linspace(0.1, 2.0, 20):
            exam_probs = 1 / (positions ** alpha)
            # Approximate log-likelihood (assuming uniform relevance)
            ll = np.sum(clicks * np.log(exam_probs + 1e-10) + 
                       (1 - clicks) * np.log(1 - exam_probs * 0.5 + 1e-10))
            
            if ll > best_ll:
                best_ll = ll
                best_alpha = alpha
        
        self.power_param = best_alpha
        self.examination_probs = 1 / (np.arange(1, self.max_position + 1) ** best_alpha)
    
    def _fit_exponential(self, positions: np.ndarray, clicks: np.ndarray) -> None:
        """Fit exponential decay bias."""
        def neg_ll(decay):
            exam_probs = np.exp(-decay * (positions - 1))
            ll = np.sum(clicks * np.log(exam_probs + 1e-10))
            return -ll
        
        result = minimize(neg_ll, x0=[0.1], bounds=[(0.01, 1.0)])
        decay = result.x[0]
        
        self.examination_probs = np.exp(-decay * np.arange(self.max_position))
    
    def _fit_em(
        self,
        positions: np.ndarray,
        clicks: np.ndarray,
        relevance_estimates: np.ndarray | None,
    ) -> None:
        """
        Fit using EM algorithm (Position-Based Model).
        
        Jointly estimates examination probabilities and relevance.
        """
        n_positions = self.max_position
        
        # Initialize
        exam_probs = 1 / np.sqrt(np.arange(1, n_positions + 1))
        
        if relevance_estimates is None:
            # Use click rate as initial relevance estimate
            relevance = np.ones(len(clicks)) * 0.5
        else:
            relevance = relevance_estimates
        
        # EM iterations
        for _ in range(10):
            # E-step: estimate examination given clicks and relevance
            pos_exam = exam_probs[positions.astype(int) - 1]
            
            # P(examine | click=1, relevance) ∝ exam_prob * relevance
            gamma = np.where(
                clicks == 1,
                1.0,  # If clicked, must have examined
                (pos_exam * (1 - relevance)) / (1 - pos_exam * relevance + 1e-10)
            )
            
            # M-step: update examination probabilities
            new_exam_probs = np.zeros(n_positions)
            for pos in range(1, n_positions + 1):
                mask = positions == pos
                if mask.sum() > 0:
                    new_exam_probs[pos - 1] = gamma[mask].mean()
            
            exam_probs = new_exam_probs
        
        self.examination_probs = exam_probs
    
    def get_propensity(self, position: int) -> float:
        """
        Get examination probability for a position.
        
        Args:
            position: 1-indexed position
            
        Returns:
            Examination probability
        """
        if self.examination_probs is None:
            # Default power law
            return 1 / (position ** self.power_param)
        
        if position > len(self.examination_probs):
            return self.examination_probs[-1]
        
        return self.examination_probs[position - 1]
    
    def get_propensities(self, positions: np.ndarray) -> np.ndarray:
        """Get propensities for multiple positions."""
        return np.array([self.get_propensity(int(p)) for p in positions])


class InversePropensityWeighting:
    """
    Inverse Propensity Weighting (IPW) for unbiased LTR.
    
    Reweights training examples to correct for position bias,
    allowing the model to learn true relevance rather than
    position-influenced click patterns.
    """
    
    def __init__(
        self,
        bias_corrector: PositionBiasCorrector | None = None,
        clip_weight: float = 10.0,
    ):
        self.bias_corrector = bias_corrector or PositionBiasCorrector()
        self.clip_weight = clip_weight
    
    def fit(
        self,
        positions: np.ndarray,
        clicks: np.ndarray,
    ) -> "InversePropensityWeighting":
        """
        Fit the bias model on observed data.
        
        Args:
            positions: Position of each impression
            clicks: Binary click indicator
            
        Returns:
            self
        """
        self.bias_corrector.fit(positions, clicks)
        return self
    
    def compute_weights(
        self,
        positions: np.ndarray,
        clicks: np.ndarray,
    ) -> np.ndarray:
        """
        Compute sample weights for training.
        
        For clicked items: weight = 1 / P(examine|position)
        For non-clicked items: weight = 1 (no correction needed)
        
        Args:
            positions: Position of each example
            clicks: Binary click indicator
            
        Returns:
            Sample weights
        """
        propensities = self.bias_corrector.get_propensities(positions)
        
        # IPW weights for clicks
        weights = np.where(
            clicks == 1,
            1 / (propensities + 1e-10),
            1.0
        )
        
        # Clip extreme weights
        weights = np.clip(weights, 1.0, self.clip_weight)
        
        # Normalize
        weights = weights / weights.mean()
        
        return weights
    
    def compute_doubly_robust_weights(
        self,
        positions: np.ndarray,
        clicks: np.ndarray,
        relevance_estimates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute doubly robust weights and adjusted labels.
        
        Combines IPW with a relevance model for lower variance.
        
        Args:
            positions: Position of each example
            clicks: Binary click indicator
            relevance_estimates: Model-based relevance predictions
            
        Returns:
            (weights, adjusted_labels)
        """
        propensities = self.bias_corrector.get_propensities(positions)
        
        # Doubly robust estimator
        ipw_term = (clicks - propensities * relevance_estimates) / (propensities + 1e-10)
        dr_labels = relevance_estimates + ipw_term
        
        # Clip labels to valid range
        dr_labels = np.clip(dr_labels, 0, 1)
        
        # Weights are simpler in DR
        weights = np.ones_like(positions, dtype=float)
        
        return weights, dr_labels


class ClickModel:
    """
    Simple click model for generating synthetic click data with position bias.
    
    Useful for testing bias correction methods.
    """
    
    def __init__(
        self,
        examination_model: Literal["cascade", "position"] = "position",
        examination_decay: float = 0.5,
    ):
        self.examination_model = examination_model
        self.examination_decay = examination_decay
    
    def simulate_clicks(
        self,
        relevance: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate clicks with position bias.
        
        Click probability = P(examine|position) * P(click|relevant, examine)
        
        Args:
            relevance: True relevance scores [0, 1]
            positions: Display positions (1-indexed)
            
        Returns:
            Binary click indicators
        """
        # Examination probability
        if self.examination_model == "cascade":
            # Users examine until they click
            exam_probs = np.cumprod(
                np.insert(1 - relevance[:-1] * 0.5, 0, 1)
            )[:len(positions)]
        else:
            # Position-based examination
            exam_probs = 1 / (positions ** self.examination_decay)
        
        # Click probability = examine * relevance
        click_probs = exam_probs * relevance
        
        # Sample clicks
        clicks = np.random.binomial(1, click_probs)
        
        return clicks
