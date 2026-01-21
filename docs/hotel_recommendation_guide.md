# Hotel Recommendation Guide

## Overview

This guide explains how CusFlow handles hotel recommendations, similar to platforms like Expedia, Booking.com, or Hotels.com.

## Hotel Features

### Core Item Features

| Feature | Type | Description |
|---------|------|-------------|
| `star_rating` | int | Hotel star rating (1-5) |
| `review_score` | float | Guest review score (1-5) |
| `price_per_night` | float | Price in USD |
| `location_score` | float | Location quality score |
| `amenities_count` | int | Number of amenities |
| `distance_to_center` | float | Distance to city center (km) |
| `booking_count` | int | Historical bookings |
| `cancellation_policy` | int | 0=strict, 1=moderate, 2=flexible |

### User Features

| Feature | Type | Description |
|---------|------|-------------|
| `booking_history_count` | int | Past bookings |
| `avg_spend` | float | Average booking value |
| `preferred_star_rating` | int | Preferred hotel class |
| `loyalty_tier` | int | Membership level |
| `days_since_last_booking` | int | Recency |

### Context Features

| Feature | Type | Description |
|---------|------|-------------|
| `check_in_date` | date | Arrival date |
| `check_out_date` | date | Departure date |
| `guests` | int | Number of guests |
| `room_type` | str | Room preference |
| `device_type` | str | web/mobile/tablet |

## Example Data

### Sample Hotels

```python
{
    "item_id": "hotel_00042",
    "name": "The Grand Plaza Hotel",
    "description": "A beautiful 5-star hotel located in Downtown...",
    "features": {
        "star_rating": 5,
        "review_score": 4.7,
        "price_per_night": 289.99,
        "location_score": 4.8,
        "amenities_count": 18,
        "distance_to_center": 0.5,
        "booking_count": 1250,
        "cancellation_policy": 2
    }
}
```

### Sample User

```python
{
    "user_id": "user_00123",
    "features": {
        "booking_history_count": 12,
        "avg_spend": 185.50,
        "preferred_star_rating": 4,
        "loyalty_tier": 2,
        "days_since_last_booking": 45
    },
    "segments": ["frequent_traveler", "business"]
}
```

## Ranking Logic

### Feature Importance (Typical)

1. **User Preference Match** - Star rating alignment
2. **Price-Value Ratio** - Price vs. user's typical spend
3. **Quality Score** - Reviews + ratings composite
4. **Popularity** - Booking momentum
5. **GenAI Embedding Similarity** - Semantic match to user preferences

### Cross Features

```python
# Price alignment
price_ratio = (price_per_night - user_avg_spend) / user_avg_spend

# Star preference match
star_match = 1 - abs(preferred_star - actual_star) / 5

# Value score
value = review_score / (price_per_night / 100)
```

## GenAI Enhancement

### Hotel Summary Prompt

```
Summarize this hotel for a traveler in 2-3 sentences.
Focus on: location highlights, standout amenities, ideal guest type, 
and value proposition.

Hotel: The Grand Plaza Hotel
Description: A beautiful 5-star hotel located in Downtown...
Features: star_rating=5, review_score=4.7, price_per_night=289.99...
```

### Generated Summary

> "The Grand Plaza Hotel is a luxurious 5-star property in the heart of 
> downtown, perfect for business travelers and couples seeking premium 
> comfort. With 18 amenities including spa and rooftop pool, and a stellar 
> 4.7 rating, it offers excellent value for the discerning traveler."

### Embedding Usage

```python
# User search: "luxury hotel with spa near business district"
user_query_embedding = embed(query)

# Compute similarity to all hotels
similarities = cosine_similarity(user_query_embedding, hotel_embeddings)

# Use as ranking feature
features["genai_embedding_sim"] = similarities[hotel_idx]
```

## A/B Test Scenarios

### Experiment: GenAI Features

**Hypothesis**: Adding LLM-generated summaries and embeddings improves CTR.

**Setup**:
- Control: Baseline ranking (no GenAI)
- Treatment: Ranking with GenAI features

**Expected Uplift**:
- NDCG@10: +3-5%
- CTR: +2-4%
- Booking Rate: +1-2%

### Experiment: Bias Correction

**Hypothesis**: IPW training improves ranking quality.

**Setup**:
- Control: Standard LambdaMART
- Treatment: LambdaMART with IPW weights

**Expected Uplift**:
- NDCG@10: +1-3%
- Position 1 conversion rate: +5%

## Sample Workflow

```bash
# 1. Generate hotel data
python -m src.cli generate-data --domain hotel --items 1000

# 2. Train model
python scripts/train_model.py --domain hotel --bias-correction

# 3. Evaluate
python scripts/evaluate.py --ablation

# 4. Run A/B simulation
python scripts/run_ab_sim.py --control popularity --treatment models/lambdamart_v1.joblib

# 5. Start API
python -m src.cli serve
```

## API Example

```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_00123",
    "candidate_ids": ["hotel_001", "hotel_002", "hotel_003"],
    "context": {
      "device_type": "mobile",
      "context": {
        "check_in_date": "2024-03-15",
        "guests": 2,
        "room_type": "double"
      }
    },
    "use_genai_features": true
  }'
```

## Metrics to Track

| Metric | Target | Description |
|--------|--------|-------------|
| NDCG@10 | >0.80 | Ranking quality |
| CTR | >5% | Click-through rate |
| Booking Rate | >2% | Conversion rate |
| Latency P99 | <100ms | Response time |
