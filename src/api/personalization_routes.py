"""
Real-Time Personalization API Endpoints

Provides endpoints for:
- Personalized recommendations (bandit-powered)
- Dynamic pricing
- Hybrid search
"""

import time
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/personalization", tags=["Personalization"])


class PersonalizationRequest(BaseModel):
    user_id: str
    context: dict[str, Any] = Field(default_factory=dict)
    num_results: int = Field(default=10, ge=1, le=100)
    use_bandit: bool = True


class PersonalizedItem(BaseModel):
    item_id: str
    score: float
    position: int
    strategy: str = "default"


class PersonalizationResponse(BaseModel):
    request_id: str
    user_id: str
    items: list[PersonalizedItem]
    strategy_used: str
    latency_ms: float


class DynamicPriceRequest(BaseModel):
    item_id: str
    user_id: str | None = None
    check_in_date: str | None = None


class DynamicPriceResponse(BaseModel):
    item_id: str
    base_price: float
    recommended_price: float
    price_multiplier: float
    recommendation_reason: str


@router.post("/recommend", response_model=PersonalizationResponse)
async def get_personalized_recommendations(request: Request, req: PersonalizationRequest) -> PersonalizationResponse:
    """Get personalized recommendations using multi-armed bandit."""
    start = time.perf_counter()
    request_id = str(uuid.uuid4())
    
    bandit = getattr(request.app.state, "personalization_bandit", None)
    strategy = "hybrid"
    
    if bandit and req.use_bandit:
        context = np.array([req.context.get("price_sensitivity", 0.5), req.context.get("loyalty", 0) / 3, 0.5, 0.5, 0.5])
        result = bandit.get_strategy(context=context, user_id=req.user_id)
        strategy = result["strategy_id"]
    
    items = [PersonalizedItem(item_id=f"hotel_{i:06d}", score=0.9-i*0.05, position=i+1, strategy=strategy) 
             for i in range(req.num_results)]
    
    return PersonalizationResponse(request_id=request_id, user_id=req.user_id, items=items,
                                   strategy_used=strategy, latency_ms=(time.perf_counter()-start)*1000)


@router.post("/price", response_model=DynamicPriceResponse)
async def get_dynamic_price(request: Request, req: DynamicPriceRequest) -> DynamicPriceResponse:
    """Get dynamically optimized price."""
    from src.pricing.dynamic_pricing import DynamicPricingEngine, PricingContext
    
    context = PricingContext(item_id=req.item_id, base_price=150.0, demand_score=0.7,
                            inventory_level=0.3, days_until_event=14)
    engine = DynamicPricingEngine()
    rec = engine.optimize_price(context)
    
    return DynamicPriceResponse(item_id=req.item_id, base_price=rec.base_price,
                               recommended_price=rec.recommended_price, price_multiplier=rec.price_multiplier,
                               recommendation_reason=rec.recommendation_reason)


@router.get("/stats")
async def get_stats(request: Request) -> dict[str, Any]:
    """Get personalization statistics."""
    bandit = getattr(request.app.state, "personalization_bandit", None)
    if not bandit:
        return {"status": "not_initialized"}
    return {"status": "active", "stats": bandit.get_stats()}
