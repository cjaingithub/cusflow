"""
Dynamic Pricing Optimization Module

Implements revenue-optimal pricing strategies for travel/e-commerce:
- Price elasticity modeling
- Demand forecasting
- Competitive positioning
- Real-time price optimization
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class PricingContext:
    """Context for pricing decision."""
    item_id: str
    base_price: float
    demand_score: float
    inventory_level: float
    days_until_event: int
    competitor_avg_price: float | None = None
    market_position: float = 0.5
    conversion_rate_7d: float = 0.05
    searches_24h: int = 0
    is_weekend: bool = False
    is_holiday: bool = False
    is_peak_season: bool = False


@dataclass
class PricingRecommendation:
    """Output of pricing optimization."""
    item_id: str
    base_price: float
    recommended_price: float
    price_multiplier: float
    expected_demand: float
    expected_revenue: float
    expected_conversion_rate: float
    confidence_interval: tuple[float, float]
    price_factors: dict[str, float]
    recommendation_reason: str


class PriceElasticityModel:
    """Models price elasticity of demand. Elasticity ≈ -1.5 to -2.5 for hotels."""
    
    def __init__(self, base_elasticity: float = -1.8):
        self.base_elasticity = base_elasticity
        self.elasticity_by_segment = {
            "budget": -2.5, "mid_range": -1.8, "premium": -1.2, "luxury": -0.8,
        }
    
    def estimate_elasticity(self, segment: str | None = None, days_until_event: int = 30, inventory_level: float = 0.5) -> float:
        elasticity = self.elasticity_by_segment.get(segment, self.base_elasticity) if segment else self.base_elasticity
        if days_until_event <= 3:
            elasticity *= 0.6
        elif days_until_event <= 7:
            elasticity *= 0.8
        elif days_until_event >= 60:
            elasticity *= 1.2
        if inventory_level < 0.1:
            elasticity *= 0.5
        elif inventory_level < 0.3:
            elasticity *= 0.7
        return elasticity
    
    def demand_curve(self, price: float, base_price: float, base_demand: float, elasticity: float) -> float:
        """Q = Q₀ × (P/P₀)^ε"""
        if base_price <= 0 or price <= 0:
            return 0
        return max(0, base_demand * (price / base_price) ** elasticity)


class DynamicPricingEngine:
    """Revenue-optimal dynamic pricing engine."""
    
    def __init__(self, min_price_multiplier: float = 0.7, max_price_multiplier: float = 2.0):
        self.elasticity_model = PriceElasticityModel()
        self.min_multiplier = min_price_multiplier
        self.max_multiplier = max_price_multiplier
    
    def optimize_price(self, context: PricingContext, segment: str | None = None) -> PricingRecommendation:
        base_price = context.base_price
        elasticity = self.elasticity_model.estimate_elasticity(segment, context.days_until_event, context.inventory_level)
        base_demand = context.demand_score
        
        def revenue(price: float) -> float:
            demand = self.elasticity_model.demand_curve(price, base_price, base_demand, elasticity)
            return -1 * price * demand
        
        min_price, max_price = base_price * self.min_multiplier, base_price * self.max_multiplier
        result = minimize_scalar(revenue, bounds=(min_price, max_price), method='bounded')
        optimal_price = result.x
        
        price_factors = self._calculate_price_factors(context)
        adjustment = sum(price_factors.values()) / len(price_factors)
        final_price = max(min_price, min(max_price, optimal_price * adjustment))
        
        expected_demand = self.elasticity_model.demand_curve(final_price, base_price, base_demand, elasticity)
        multiplier = final_price / base_price
        
        return PricingRecommendation(
            item_id=context.item_id, base_price=base_price,
            recommended_price=round(final_price, 2), price_multiplier=round(multiplier, 3),
            expected_demand=round(expected_demand, 4), expected_revenue=round(final_price * expected_demand, 2),
            expected_conversion_rate=round(context.conversion_rate_7d * (base_price / final_price) ** abs(elasticity), 4),
            confidence_interval=(round(final_price * 0.9, 2), round(final_price * 1.1, 2)),
            price_factors=price_factors, recommendation_reason=self._generate_reason(context, multiplier),
        )
    
    def _calculate_price_factors(self, context: PricingContext) -> dict[str, float]:
        factors = {}
        factors["scarcity"] = 1.4 if context.inventory_level < 0.1 else (1.2 if context.inventory_level < 0.3 else 1.0)
        factors["urgency"] = 1.3 if context.days_until_event <= 3 else (1.15 if context.days_until_event <= 7 else 1.0)
        factors["demand"] = 1.2 if context.demand_score > 0.8 else (0.85 if context.demand_score < 0.3 else 1.0)
        factors["seasonality"] = 1.35 if context.is_holiday else (1.25 if context.is_peak_season else (1.1 if context.is_weekend else 1.0))
        return factors
    
    def _generate_reason(self, context: PricingContext, multiplier: float) -> str:
        if multiplier > 1.2:
            reasons = []
            if context.inventory_level < 0.2: reasons.append("low inventory")
            if context.days_until_event <= 7: reasons.append("approaching date")
            if context.demand_score > 0.7: reasons.append("high demand")
            return f"Price increase: {', '.join(reasons) or 'market conditions'}"
        elif multiplier < 0.9:
            reasons = []
            if context.inventory_level > 0.7: reasons.append("high inventory")
            if context.demand_score < 0.3: reasons.append("low demand")
            return f"Price decrease: {', '.join(reasons) or 'market conditions'}"
        return "Price near optimal for current conditions"
