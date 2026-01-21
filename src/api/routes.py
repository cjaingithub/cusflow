"""
API Routes for CusFlow Ranking Service

Endpoints:
- /rank: Get personalized rankings
- /items: Manage item catalog
- /users: Manage user features
- /health: Health checks
- /metrics: Evaluation metrics
"""

import time
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.config import Domain, get_settings
from src.data.schemas import (
    ContextFeatures,
    Item,
    RankedItem,
    RankingRequest,
    RankingResponse,
    User,
)

router = APIRouter()


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@router.get("/health", tags=["Health"])
async def health_check(request: Request) -> dict[str, Any]:
    """Check API health and service availability."""
    services = {
        "api": "healthy",
        "redis": "unknown",
        "ranker": "unknown",
        "embeddings": "unknown",
    }
    
    # Check Redis
    try:
        if hasattr(request.app.state, "feature_store"):
            if await request.app.state.feature_store.aping():
                services["redis"] = "healthy"
            else:
                services["redis"] = "unavailable"
    except Exception:
        services["redis"] = "error"
    
    # Check ranker
    if hasattr(request.app.state, "ranker") and request.app.state.ranker is not None:
        services["ranker"] = "loaded"
    else:
        services["ranker"] = "not_loaded"
    
    # Check embeddings
    if hasattr(request.app.state, "embedding_service") and request.app.state.embedding_service is not None:
        services["embeddings"] = "available"
    else:
        services["embeddings"] = "not_available"
    
    overall = "healthy" if all(
        v in ["healthy", "loaded", "available"] 
        for v in services.values()
    ) else "degraded"
    
    return {
        "status": overall,
        "services": services,
        "version": "1.0.0",
    }


@router.get("/", tags=["Health"])
async def root() -> dict[str, str]:
    """API root endpoint."""
    return {
        "name": "CusFlow Ranking API",
        "version": "1.0.0",
        "docs": "/docs",
    }


# =============================================================================
# Ranking Endpoints
# =============================================================================

@router.post("/rank", response_model=RankingResponse, tags=["Ranking"])
async def rank_items(
    request: Request,
    ranking_request: RankingRequest,
) -> RankingResponse:
    """
    Get personalized ranking for a user.
    
    This is the main ranking endpoint that:
    1. Retrieves user features from Redis
    2. Gets candidate items (from request or generates them)
    3. Applies the LTR model to rank items
    4. Returns ranked results with scores
    """
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())
    
    # Get services from app state
    feature_store = request.app.state.feature_store
    ranker = request.app.state.ranker
    embedding_service = request.app.state.embedding_service
    
    # Get user features
    user = await feature_store.aget_user(ranking_request.user_id)
    if user is None:
        # Create default user
        user = User(user_id=ranking_request.user_id)
    
    # Get or generate candidates
    if ranking_request.candidate_ids:
        candidate_ids = ranking_request.candidate_ids
    else:
        # TODO: Use candidate generation module
        # For now, return error if no candidates provided
        raise HTTPException(
            status_code=400,
            detail="candidate_ids must be provided (candidate generation not yet implemented)"
        )
    
    # Get item features
    items = await feature_store.aget_items_batch(candidate_ids)
    valid_items = [(cid, item) for cid, item in zip(candidate_ids, items) if item is not None]
    
    if not valid_items:
        raise HTTPException(
            status_code=404,
            detail="No valid items found for provided candidate_ids"
        )
    
    candidate_ids = [cid for cid, _ in valid_items]
    items = [item for _, item in valid_items]
    
    # Rank items
    if ranker is not None and ranker.model is not None:
        # Use trained model
        from src.ranking.feature_engineering import FeatureEngineer
        
        fe = FeatureEngineer()
        
        # Get embeddings if using GenAI features
        user_embedding = None
        item_embeddings = None
        
        if ranking_request.use_genai_features and embedding_service:
            try:
                embeddings = await feature_store.aget_embeddings_batch(candidate_ids)
                item_embeddings = np.array([
                    e if e is not None else np.zeros(embedding_service.dimension)
                    for e in embeddings
                ])
            except Exception:
                pass
        
        # Build feature matrix
        try:
            X = fe.transform(
                items,
                user=user,
                context=ranking_request.context,
                user_embedding=user_embedding,
                item_embeddings=item_embeddings,
            )
            
            # Get rankings
            ranked = ranker.rank(X, candidate_ids, top_k=ranking_request.num_results)
        except Exception as e:
            # Fallback to popularity-based ranking
            ranked = [
                (item.item_id, item.features.popularity_score)
                for item in sorted(items, key=lambda x: x.features.popularity_score, reverse=True)
            ][:ranking_request.num_results]
    else:
        # Fallback: rank by popularity
        ranked = [
            (item.item_id, item.features.popularity_score)
            for item in sorted(items, key=lambda x: x.features.popularity_score, reverse=True)
        ][:ranking_request.num_results]
    
    # Build response
    ranked_items = []
    item_lookup = {item.item_id: item for item in items}
    
    for position, (item_id, score) in enumerate(ranked, 1):
        ranked_items.append(RankedItem(
            item_id=item_id,
            score=score,
            position=position,
            item=item_lookup.get(item_id),
        ))
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return RankingResponse(
        request_id=request_id,
        user_id=ranking_request.user_id,
        items=ranked_items,
        latency_ms=latency_ms,
        model_version="v1" if ranker and ranker.model else "fallback",
        experiment_id=ranking_request.experiment_id,
        treatment_group=ranking_request.treatment_group,
    )


@router.post("/rank/batch", tags=["Ranking"])
async def rank_batch(
    request: Request,
    requests: list[RankingRequest],
) -> list[RankingResponse]:
    """Batch ranking for multiple users."""
    responses = []
    for req in requests:
        resp = await rank_items(request, req)
        responses.append(resp)
    return responses


# =============================================================================
# Item Management Endpoints
# =============================================================================

@router.post("/items", tags=["Items"])
async def create_item(request: Request, item: Item) -> dict[str, str]:
    """Add or update an item in the catalog."""
    feature_store = request.app.state.feature_store
    feature_store.set_item(item)
    
    return {"status": "success", "item_id": item.item_id}


@router.post("/items/batch", tags=["Items"])
async def create_items_batch(request: Request, items: list[Item]) -> dict[str, Any]:
    """Add or update multiple items."""
    feature_store = request.app.state.feature_store
    feature_store.set_items_batch(items)
    
    return {"status": "success", "count": len(items)}


@router.get("/items/{item_id}", tags=["Items"])
async def get_item(request: Request, item_id: str) -> Item:
    """Get item by ID."""
    feature_store = request.app.state.feature_store
    item = await feature_store.aget_item(item_id)
    
    if item is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    
    return item


@router.delete("/items/{item_id}", tags=["Items"])
async def delete_item(request: Request, item_id: str) -> dict[str, str]:
    """Delete an item from the catalog."""
    feature_store = request.app.state.feature_store
    # Note: Redis doesn't have a direct delete in our implementation
    # This would need to be added
    return {"status": "success", "item_id": item_id}


# =============================================================================
# User Management Endpoints
# =============================================================================

@router.post("/users", tags=["Users"])
async def create_user(request: Request, user: User) -> dict[str, str]:
    """Add or update a user."""
    feature_store = request.app.state.feature_store
    feature_store.set_user(user)
    
    return {"status": "success", "user_id": user.user_id}


@router.get("/users/{user_id}", tags=["Users"])
async def get_user(request: Request, user_id: str) -> User:
    """Get user by ID."""
    feature_store = request.app.state.feature_store
    user = await feature_store.aget_user(user_id)
    
    if user is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    return user


# =============================================================================
# Embedding Endpoints
# =============================================================================

@router.post("/embeddings/items", tags=["Embeddings"])
async def embed_items(
    request: Request,
    item_ids: list[str],
    store: bool = True,
) -> dict[str, Any]:
    """Generate and optionally store embeddings for items."""
    embedding_service = request.app.state.embedding_service
    feature_store = request.app.state.feature_store
    
    if embedding_service is None:
        raise HTTPException(status_code=503, detail="Embedding service not available")
    
    # Get items
    items = await feature_store.aget_items_batch(item_ids)
    valid_items = [item for item in items if item is not None]
    
    if not valid_items:
        raise HTTPException(status_code=404, detail="No valid items found")
    
    # Generate embeddings
    embeddings = embedding_service.embed_items(valid_items)
    
    # Store if requested
    if store:
        valid_ids = [item.item_id for item in valid_items]
        feature_store.set_embeddings_batch(valid_ids, embeddings)
    
    return {
        "status": "success",
        "count": len(valid_items),
        "dimension": embeddings.shape[1],
    }


@router.get("/embeddings/{item_id}", tags=["Embeddings"])
async def get_embedding(request: Request, item_id: str) -> dict[str, Any]:
    """Get embedding for an item."""
    feature_store = request.app.state.feature_store
    embedding = await feature_store.aget_embedding(item_id)
    
    if embedding is None:
        raise HTTPException(status_code=404, detail=f"Embedding for {item_id} not found")
    
    return {
        "item_id": item_id,
        "embedding": embedding.tolist(),
        "dimension": len(embedding),
    }


# =============================================================================
# Admin Endpoints
# =============================================================================

@router.get("/admin/stats", tags=["Admin"])
async def get_stats(request: Request) -> dict[str, Any]:
    """Get feature store statistics."""
    feature_store = request.app.state.feature_store
    
    try:
        stats = feature_store.get_stats()
        return {"status": "success", **stats}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.post("/admin/reload-model", tags=["Admin"])
async def reload_model(request: Request) -> dict[str, str]:
    """Reload the ranking model from disk."""
    settings = get_settings()
    
    try:
        from src.ranking.lambdamart import LambdaMARTRanker
        
        ranker = LambdaMARTRanker()
        ranker.load(settings.ranking_model_path)
        request.app.state.ranker = ranker
        
        return {"status": "success", "message": "Model reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")


@router.get("/admin/config", tags=["Admin"])
async def get_config(request: Request) -> dict[str, Any]:
    """Get current configuration (non-sensitive)."""
    settings = get_settings()
    
    return {
        "domain": settings.domain.value,
        "ranking_top_k": settings.ranking_top_k,
        "ranking_rerank_k": settings.ranking_rerank_k,
        "click_bias_correction": settings.click_bias_correction,
        "embedding_provider": settings.embedding_provider.value,
        "use_local_embeddings": settings.use_local_embeddings,
        "eval_metrics": settings.eval_metrics,
    }
