# CusFlow Technical Guide

## Learning to Rank (LTR)

### LambdaMART Overview

LambdaMART combines:
- **Lambda gradients**: Optimizes ranking metrics directly (NDCG)
- **MART**: Multiple Additive Regression Trees (gradient boosting)

### Feature Engineering

#### Item Features
```python
# Numeric features
"star_rating", "review_score", "price_per_night"

# Categorical features (one-hot encoded)
"cancellation_policy", "room_type"

# Computed features
"popularity_score", "freshness_score", "quality_score"
```

#### Cross Features
```python
# User-item interactions
"price_vs_user_avg_spend"  # (item_price - user_avg) / user_avg
"rating_preference_match"  # 1 - |preferred - actual| / max
```

#### GenAI Features
```python
# Embedding similarity
"genai_embedding_sim"  # cosine(user_emb, item_emb)

# Summary-based
"genai_summary_length"  # Proxy for content richness
```

### Click Bias Correction

#### Position Bias Model
```
P(click | item, position) = P(examine | position) × P(click | examine, item)
```

#### Inverse Propensity Weighting
```python
propensity = 1 / (position ** decay)  # Power law
weight = 1 / propensity  # For clicked items
```

## GenAI Integration

### Embedding Generation

#### Local (sentence-transformers)
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)  # 384-dim vectors
```

#### OpenAI
```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)
# 1536-dim vectors
```

### Item Summarization

Domain-specific prompts generate concise summaries:

```python
# Hotel prompt
"Summarize this hotel for a traveler in 2-3 sentences.
Focus on: location highlights, standout amenities, ideal guest type."

# Wealth report prompt
"Summarize this report for an investor in 2-3 sentences.
Focus on: key thesis, target profile, risk considerations."
```

## Evaluation Metrics

### Ranking Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| NDCG@K | DCG / IDCG | 0-1, higher is better |
| MAP | Mean(AP) | Average precision across queries |
| Recall@K | Relevant in K / Total Relevant | Coverage |
| MRR | 1 / First Relevant Position | Speed to first hit |

### A/B Simulation

```python
# Simulate clicks with position bias
P(click) = P(examine|pos) × P(click|examine, relevance)

# Metrics
CTR = clicks / impressions
CVR = conversions / clicks

# Uplift
relative_uplift = (treatment - control) / control × 100%
```

### Statistical Significance

Two-sample t-test for each metric:
```python
t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
is_significant = p_value < 0.05
```

## API Design

### Request/Response Pattern

```python
# Request
{
    "user_id": "user_001",
    "candidate_ids": ["item_1", "item_2"],
    "context": {"device_type": "mobile"},
    "use_genai_features": true
}

# Response
{
    "request_id": "uuid",
    "items": [
        {"item_id": "item_1", "score": 0.95, "position": 1},
        {"item_id": "item_2", "score": 0.82, "position": 2}
    ],
    "latency_ms": 15.3
}
```

### Async Operations

All Redis operations support async for non-blocking I/O:

```python
async def rank_items(request: RankingRequest):
    user = await feature_store.aget_user(request.user_id)
    items = await feature_store.aget_items_batch(request.candidate_ids)
    # ...
```

## Performance Optimization

### Inference Latency

| Component | Target | Technique |
|-----------|--------|-----------|
| Feature lookup | <5ms | Redis pipeline |
| Feature engineering | <2ms | Vectorized NumPy |
| Model scoring | <5ms | LightGBM native |
| Total P99 | <50ms | Async + caching |

### Embedding Caching

```python
# Cache embeddings in Redis
store.set_embeddings_batch(item_ids, embeddings)

# Retrieve on inference
embeddings = await store.aget_embeddings_batch(candidate_ids)
```

## Deployment

### Docker Images

```dockerfile
# Multi-stage build
FROM python:3.11-slim as builder
# Install deps

FROM python:3.11-slim as runtime
# Copy from builder
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0"]
```

### Health Checks

```python
@router.get("/health")
async def health_check():
    return {
        "api": "healthy",
        "redis": await check_redis(),
        "model": check_model_loaded()
    }
```

## Monitoring

### Key Metrics to Track

1. **Latency**: P50, P95, P99 response times
2. **Throughput**: Requests per second
3. **Error rate**: 4xx, 5xx responses
4. **Model metrics**: NDCG, CTR in production
5. **Cache hit rate**: Redis cache efficiency
