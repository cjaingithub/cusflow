"""
A/B Test Simulation for Ranking Systems.

Replay historical traffic to simulate online experiments
and estimate uplift between control and treatment models.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol

import numpy as np
from tqdm import tqdm

from src.data.schemas import ABTestResult, ContextFeatures, FeedbackEvent, User
from src.evaluation.metrics import RankingMetrics, compute_lift


class RankerProtocol(Protocol):
    """Protocol for ranking models."""
    
    def rank(
        self,
        user: User,
        candidate_ids: list[str],
        context: ContextFeatures | None = None,
    ) -> list[tuple[str, float]]:
        """Rank candidates and return (item_id, score) tuples."""
        ...


class TrafficSplitter:
    """
    Split traffic between control and treatment groups.
    
    Uses consistent hashing for deterministic assignment.
    """
    
    def __init__(
        self,
        treatment_ratio: float = 0.5,
        salt: str = "cusflow_ab",
    ):
        self.treatment_ratio = treatment_ratio
        self.salt = salt
    
    def assign_group(self, user_id: str) -> str:
        """
        Assign user to control or treatment group.
        
        Uses hash-based assignment for consistency.
        
        Args:
            user_id: User identifier
            
        Returns:
            "control" or "treatment"
        """
        hash_input = f"{self.salt}:{user_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        
        if (hash_value % 1000) / 1000 < self.treatment_ratio:
            return "treatment"
        return "control"
    
    def split_users(self, user_ids: list[str]) -> tuple[list[str], list[str]]:
        """
        Split users into control and treatment groups.
        
        Returns:
            (control_user_ids, treatment_user_ids)
        """
        control = []
        treatment = []
        
        for user_id in user_ids:
            if self.assign_group(user_id) == "treatment":
                treatment.append(user_id)
            else:
                control.append(user_id)
        
        return control, treatment


@dataclass
class SimulationConfig:
    """Configuration for A/B simulation."""
    
    experiment_id: str = "default"
    treatment_ratio: float = 0.5
    
    # Click model parameters
    position_bias_decay: float = 0.5  # P(examine) = 1/position^decay
    click_noise: float = 0.1  # Random click probability
    
    # Conversion model
    conversion_rate: float = 0.1  # Base conversion rate
    relevance_boost: float = 2.0  # Boost for highly relevant items
    
    # Metrics to compute
    metrics: list[str] = field(default_factory=lambda: ["ndcg@10", "ctr", "cvr"])


class ABSimulator:
    """
    Simulate A/B test by replaying historical traffic.
    
    Features:
    - Click model with position bias
    - Conversion simulation
    - CTR/CVR estimation
    - Statistical significance testing
    """
    
    def __init__(
        self,
        control_ranker: Any,
        treatment_ranker: Any,
        config: SimulationConfig | None = None,
    ):
        self.control_ranker = control_ranker
        self.treatment_ranker = treatment_ranker
        self.config = config or SimulationConfig()
        
        self.splitter = TrafficSplitter(self.config.treatment_ratio)
        self.metrics_calculator = RankingMetrics()
        
        # Results storage
        self.control_results: list[dict] = []
        self.treatment_results: list[dict] = []
    
    def simulate(
        self,
        events: list[FeedbackEvent],
        users: dict[str, User],
        items: dict[str, Any],
        show_progress: bool = True,
    ) -> ABTestResult:
        """
        Run A/B simulation on historical events.
        
        Args:
            events: Historical feedback events
            users: User ID -> User mapping
            items: Item ID -> Item mapping
            show_progress: Show progress bar
            
        Returns:
            ABTestResult with metrics and uplift
        """
        self.control_results = []
        self.treatment_results = []
        
        # Group events by user session
        sessions = self._group_into_sessions(events)
        
        iterator = tqdm(sessions, desc="Simulating") if show_progress else sessions
        
        for session in iterator:
            user_id = session[0].user_id
            user = users.get(user_id)
            
            if user is None:
                continue
            
            # Determine group
            group = self.splitter.assign_group(user_id)
            
            # Get candidate items from session
            candidate_ids = list(set(e.item_id for e in session))
            
            if len(candidate_ids) < 2:
                continue
            
            # Get ground truth relevance from events
            relevance = self._compute_relevance(session, candidate_ids)
            
            # Rank with appropriate model
            ranker = self.treatment_ranker if group == "treatment" else self.control_ranker
            
            try:
                ranked = ranker.rank(user, candidate_ids)
                ranked_ids = [item_id for item_id, _ in ranked]
            except Exception:
                # Fallback to random ranking
                ranked_ids = candidate_ids.copy()
                np.random.shuffle(ranked_ids)
            
            # Simulate clicks based on ranking
            clicks = self._simulate_clicks(ranked_ids, relevance)
            conversions = self._simulate_conversions(ranked_ids, relevance, clicks)
            
            # Compute metrics
            y_true = np.array([relevance.get(item_id, 0) for item_id in ranked_ids])
            y_pred = np.arange(len(ranked_ids), 0, -1).astype(float)  # Position-based scores
            
            metrics = self.metrics_calculator.evaluate(y_true, y_pred)
            metrics["ctr"] = np.mean(clicks)
            metrics["cvr"] = np.mean(conversions)
            metrics["clicks"] = int(np.sum(clicks))
            metrics["conversions"] = int(np.sum(conversions))
            
            if group == "treatment":
                self.treatment_results.append(metrics)
            else:
                self.control_results.append(metrics)
        
        return self._compute_final_results()
    
    def _group_into_sessions(
        self,
        events: list[FeedbackEvent],
    ) -> list[list[FeedbackEvent]]:
        """Group events into user sessions."""
        from collections import defaultdict
        
        user_sessions: dict[str, list[FeedbackEvent]] = defaultdict(list)
        
        for event in events:
            user_sessions[event.user_id].append(event)
        
        return list(user_sessions.values())
    
    def _compute_relevance(
        self,
        session: list[FeedbackEvent],
        candidate_ids: list[str],
    ) -> dict[str, float]:
        """Compute relevance from session events."""
        relevance: dict[str, float] = {item_id: 0.0 for item_id in candidate_ids}
        
        for event in session:
            if event.item_id not in relevance:
                continue
            
            # Update relevance based on event type
            if event.event_type == "purchase":
                relevance[event.item_id] = max(relevance[event.item_id], 4.0)
            elif event.event_type == "add_to_cart":
                relevance[event.item_id] = max(relevance[event.item_id], 3.0)
            elif event.event_type == "click":
                relevance[event.item_id] = max(relevance[event.item_id], 2.0)
            elif event.event_type == "impression":
                relevance[event.item_id] = max(relevance[event.item_id], 1.0)
            elif event.relevance_label is not None:
                relevance[event.item_id] = max(relevance[event.item_id], event.relevance_label)
        
        return relevance
    
    def _simulate_clicks(
        self,
        ranked_ids: list[str],
        relevance: dict[str, float],
    ) -> np.ndarray:
        """
        Simulate clicks with position bias.
        
        P(click) = P(examine|position) * P(click|examine, relevance)
        """
        clicks = np.zeros(len(ranked_ids))
        
        for i, item_id in enumerate(ranked_ids):
            position = i + 1
            
            # Position bias
            examine_prob = 1 / (position ** self.config.position_bias_decay)
            
            # Relevance-based click probability
            rel = relevance.get(item_id, 0)
            click_prob_if_examined = min(1.0, rel / 4 + self.config.click_noise)
            
            # Combined probability
            click_prob = examine_prob * click_prob_if_examined
            
            if np.random.random() < click_prob:
                clicks[i] = 1
        
        return clicks
    
    def _simulate_conversions(
        self,
        ranked_ids: list[str],
        relevance: dict[str, float],
        clicks: np.ndarray,
    ) -> np.ndarray:
        """Simulate conversions (only for clicked items)."""
        conversions = np.zeros(len(ranked_ids))
        
        for i, item_id in enumerate(ranked_ids):
            if clicks[i] == 0:
                continue
            
            rel = relevance.get(item_id, 0)
            
            # Higher relevance -> higher conversion
            cvr = self.config.conversion_rate
            if rel >= 3:
                cvr *= self.config.relevance_boost
            
            if np.random.random() < cvr:
                conversions[i] = 1
        
        return conversions
    
    def _compute_final_results(self) -> ABTestResult:
        """Aggregate results and compute uplift."""
        control_agg = self._aggregate_metrics(self.control_results)
        treatment_agg = self._aggregate_metrics(self.treatment_results)
        
        # Compute uplift
        uplift = {}
        for metric in control_agg:
            if control_agg[metric] > 0:
                uplift[metric] = compute_lift(control_agg[metric], treatment_agg[metric])
            else:
                uplift[metric] = 0.0
        
        # Statistical significance (t-test)
        p_values, is_significant = self._compute_significance(
            self.control_results,
            self.treatment_results,
        )
        
        return ABTestResult(
            experiment_id=self.config.experiment_id,
            control_model="control",
            treatment_model="treatment",
            control_traffic=len(self.control_results),
            treatment_traffic=len(self.treatment_results),
            control_metrics=control_agg,
            treatment_metrics=treatment_agg,
            relative_uplift=uplift,
            p_values=p_values,
            is_significant=is_significant,
        )
    
    def _aggregate_metrics(self, results: list[dict]) -> dict[str, float]:
        """Aggregate metrics across sessions."""
        if not results:
            return {}
        
        aggregated = {}
        for key in results[0]:
            values = [r[key] for r in results]
            aggregated[key] = float(np.mean(values))
        
        return aggregated
    
    def _compute_significance(
        self,
        control_results: list[dict],
        treatment_results: list[dict],
        alpha: float = 0.05,
    ) -> tuple[dict[str, float], dict[str, bool]]:
        """Compute statistical significance using t-test."""
        from scipy import stats
        
        if not control_results or not treatment_results:
            return {}, {}
        
        p_values = {}
        is_significant = {}
        
        for metric in control_results[0]:
            control_values = [r[metric] for r in control_results]
            treatment_values = [r[metric] for r in treatment_results]
            
            try:
                _, p_value = stats.ttest_ind(control_values, treatment_values)
                p_values[metric] = float(p_value)
                is_significant[metric] = p_value < alpha
            except Exception:
                p_values[metric] = 1.0
                is_significant[metric] = False
        
        return p_values, is_significant
    
    def generate_report(self, result: ABTestResult) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            f"A/B Test Report: {result.experiment_id}",
            "=" * 60,
            "",
            f"Control Traffic: {result.control_traffic}",
            f"Treatment Traffic: {result.treatment_traffic}",
            "",
            "Metrics Comparison:",
            "-" * 40,
        ]
        
        for metric in result.control_metrics:
            control = result.control_metrics[metric]
            treatment = result.treatment_metrics[metric]
            uplift = result.relative_uplift.get(metric, 0)
            sig = "✓" if result.is_significant.get(metric, False) else ""
            
            lines.append(
                f"  {metric:15s}: {control:.4f} -> {treatment:.4f} "
                f"({uplift:+.2f}%) {sig}"
            )
        
        lines.extend([
            "",
            "Statistical Significance (α=0.05):",
            "-" * 40,
        ])
        
        for metric, is_sig in result.is_significant.items():
            p_val = result.p_values.get(metric, 1.0)
            status = "SIGNIFICANT" if is_sig else "not significant"
            lines.append(f"  {metric:15s}: p={p_val:.4f} ({status})")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
