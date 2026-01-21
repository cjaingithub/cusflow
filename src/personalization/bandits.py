"""
Multi-Armed Bandit Algorithms for Personalization

Implements exploration-exploitation strategies:
- Thompson Sampling (Bayesian)
- Upper Confidence Bound (UCB)
- LinUCB (contextual bandit)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import beta


@dataclass
class ArmStats:
    """Statistics for a single arm."""
    arm_id: str
    pulls: int = 0
    successes: float = 0.0
    failures: float = 0.0
    reward_sum: float = 0.0
    reward_sum_sq: float = 0.0
    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    
    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.pulls if self.pulls > 0 else 0.0
    
    @property
    def alpha(self) -> float:
        return self.alpha_prior + self.successes
    
    @property
    def beta(self) -> float:
        return self.beta_prior + self.failures


@dataclass
class BanditResult:
    """Result of arm selection."""
    selected_arm: str
    exploration_bonus: float
    expected_reward: float
    confidence_interval: tuple[float, float]
    all_arm_scores: dict[str, float]


class BaseBandit(ABC):
    """Base class for multi-armed bandit algorithms."""
    
    def __init__(self, arm_ids: list[str]):
        self.arms = {arm_id: ArmStats(arm_id=arm_id) for arm_id in arm_ids}
    
    @abstractmethod
    def select_arm(self) -> BanditResult:
        pass
    
    def update(self, arm_id: str, reward: float, is_binary: bool = True) -> None:
        arm = self.arms[arm_id]
        arm.pulls += 1
        arm.reward_sum += reward
        arm.reward_sum_sq += reward ** 2
        if is_binary:
            if reward > 0:
                arm.successes += 1
            else:
                arm.failures += 1
    
    def get_arm_stats(self) -> dict[str, dict[str, float]]:
        return {aid: {"pulls": a.pulls, "mean_reward": a.mean_reward, "successes": a.successes} for aid, a in self.arms.items()}


class ThompsonSampling(BaseBandit):
    """Thompson Sampling using Beta-Bernoulli model."""
    
    def select_arm(self) -> BanditResult:
        samples = {aid: np.random.beta(a.alpha, a.beta) for aid, a in self.arms.items()}
        selected = max(samples, key=samples.get)
        stats = self.arms[selected]
        lower = beta.ppf(0.025, stats.alpha, stats.beta)
        upper = beta.ppf(0.975, stats.alpha, stats.beta)
        return BanditResult(selected, samples[selected] - stats.mean_reward, stats.mean_reward, (lower, upper), samples)


class UCB1(BaseBandit):
    """Upper Confidence Bound algorithm."""
    
    def __init__(self, arm_ids: list[str], exploration_coef: float = np.sqrt(2)):
        super().__init__(arm_ids)
        self.c = exploration_coef
    
    def select_arm(self) -> BanditResult:
        total = sum(a.pulls for a in self.arms.values())
        ucb = {}
        for aid, arm in self.arms.items():
            if arm.pulls == 0:
                ucb[aid] = float('inf')
            else:
                ucb[aid] = arm.mean_reward + self.c * np.sqrt(np.log(total + 1) / arm.pulls)
        selected = max(ucb, key=ucb.get)
        stats = self.arms[selected]
        width = np.sqrt(np.log(2 / 0.05) / (2 * stats.pulls)) if stats.pulls > 0 else 0.5
        return BanditResult(selected, ucb[selected] - stats.mean_reward, stats.mean_reward, 
                           (max(0, stats.mean_reward - width), min(1, stats.mean_reward + width)), ucb)


@dataclass
class ContextualArmStats:
    """Stats for contextual bandit arm."""
    arm_id: str
    dimension: int
    A: np.ndarray = field(default=None)
    b: np.ndarray = field(default=None)
    pulls: int = 0
    
    def __post_init__(self):
        if self.A is None:
            self.A = np.eye(self.dimension)
        if self.b is None:
            self.b = np.zeros((self.dimension, 1))
    
    @property
    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)


class LinUCB:
    """Linear UCB contextual bandit."""
    
    def __init__(self, arm_ids: list[str], context_dimension: int, alpha: float = 1.0):
        self.dimension = context_dimension
        self.alpha = alpha
        self.arms = {aid: ContextualArmStats(arm_id=aid, dimension=context_dimension) for aid in arm_ids}
    
    def select_arm(self, context: np.ndarray) -> BanditResult:
        context = context.reshape(-1, 1)
        ucb = {}
        for aid, arm in self.arms.items():
            theta = arm.theta
            expected = float(theta.T @ context)
            A_inv = np.linalg.inv(arm.A)
            width = self.alpha * float(np.sqrt(context.T @ A_inv @ context))
            ucb[aid] = expected + width
        selected = max(ucb, key=ucb.get)
        stats = self.arms[selected]
        exp = float(stats.theta.T @ context)
        A_inv = np.linalg.inv(stats.A)
        w = self.alpha * float(np.sqrt(context.T @ A_inv @ context))
        return BanditResult(selected, w, exp, (exp - w, exp + w), ucb)
    
    def update(self, arm_id: str, context: np.ndarray, reward: float) -> None:
        context = context.reshape(-1, 1)
        arm = self.arms[arm_id]
        arm.A = arm.A + context @ context.T
        arm.b = arm.b + reward * context
        arm.pulls += 1
    
    def get_arm_weights(self) -> dict[str, np.ndarray]:
        return {aid: arm.theta.flatten() for aid, arm in self.arms.items()}


class PersonalizationBandit:
    """High-level personalization system using bandits."""
    
    def __init__(self, strategies: list[str], context_features: list[str] | None = None, algorithm: str = "thompson_sampling", exploration_rate: float = 0.1):
        self.strategies = strategies
        self.context_features = context_features
        if algorithm == "thompson_sampling":
            self.bandit = ThompsonSampling(strategies)
        elif algorithm == "ucb":
            self.bandit = UCB1(strategies)
        elif algorithm == "linucb" and context_features:
            self.bandit = LinUCB(strategies, len(context_features))
        else:
            self.bandit = ThompsonSampling(strategies)
        self.decision_history: list[dict] = []
    
    def get_strategy(self, context: np.ndarray | None = None, user_id: str | None = None) -> dict[str, Any]:
        if isinstance(self.bandit, LinUCB) and context is not None:
            result = self.bandit.select_arm(context)
        else:
            result = self.bandit.select_arm()
        decision = {"strategy_id": result.selected_arm, "expected_reward": result.expected_reward,
                   "confidence_interval": result.confidence_interval, "exploration_bonus": result.exploration_bonus,
                   "user_id": user_id, "all_scores": result.all_arm_scores}
        self.decision_history.append(decision)
        return decision
    
    def record_outcome(self, strategy_id: str, reward: float, context: np.ndarray | None = None) -> None:
        if isinstance(self.bandit, LinUCB) and context is not None:
            self.bandit.update(strategy_id, context, reward)
        else:
            self.bandit.update(strategy_id, reward)
    
    def get_stats(self) -> dict[str, Any]:
        if isinstance(self.bandit, LinUCB):
            return {aid: {"pulls": a.pulls, "weights": self.bandit.get_arm_weights()[aid].tolist()} for aid, a in self.bandit.arms.items()}
        return self.bandit.get_arm_stats()
    
    def get_best_strategy(self) -> str:
        stats = self.get_stats()
        return max(stats.keys(), key=lambda x: stats[x].get("mean_reward", 0))
