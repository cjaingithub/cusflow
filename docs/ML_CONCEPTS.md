# Machine Learning Concepts in CusFlow

A comprehensive guide to the ML algorithms, features, and mathematical foundations used in this recommendation system.

## Table of Contents

1. [Learning to Rank (LTR)](#1-learning-to-rank-ltr)
2. [LambdaMART Algorithm](#2-lambdamart-algorithm)
3. [Why LightGBM/XGBoost vs Other Options](#3-why-lightgbmxgboost-vs-other-options)
4. [Feature Engineering](#4-feature-engineering)
5. [Approximate Nearest Neighbors (ANN) vs KNN](#5-approximate-nearest-neighbors-ann-vs-knn)
6. [Embeddings & Similarity](#6-embeddings--similarity)
7. [Click Bias Correction](#7-click-bias-correction)
8. [Evaluation Metrics](#8-evaluation-metrics)

---

## 1. Learning to Rank (LTR)

### What is Learning to Rank?

Learning to Rank is a supervised ML approach for ranking items. Unlike classification (predicting categories) or regression (predicting values), LTR optimizes the **order** of items.

### Three LTR Approaches

| Approach | Unit | Loss Function | Example |
|----------|------|---------------|---------|
| **Pointwise** | Single item | MSE, Cross-entropy | Predict relevance score per item |
| **Pairwise** | Item pairs | Hinge loss | Predict which item is better |
| **Listwise** | Entire list | NDCG-based | Optimize ranking metric directly |

### Why Listwise (LambdaMART)?

**CusFlow uses Listwise** because:

1. **Direct metric optimization**: Optimizes NDCG directly
2. **Position-aware**: Considers entire ranking, not just pairs
3. **Better correlation**: Higher offline metrics correlate with online performance

```
Pointwise: Loss = Σ (predicted_score - true_relevance)²
Pairwise:  Loss = Σ max(0, 1 - (score_i - score_j)) for all pairs where rel_i > rel_j
Listwise:  Loss = -NDCG(predicted_ranking, true_ranking)
```

---

## 2. LambdaMART Algorithm

### Overview

LambdaMART = **Lambda** gradients + **MART** (Multiple Additive Regression Trees)

It's the state-of-the-art gradient boosting algorithm for ranking, developed at Microsoft.

### The Lambda Gradient

The key innovation is computing gradients that directly optimize ranking metrics:

#### Standard Gradient (Doesn't work for ranking)
```
∂Loss/∂s_i = predicted_i - true_i
```
This doesn't account for position or relative ordering.

#### Lambda Gradient (LambdaMART)
```
λ_ij = -σ × (1 / (1 + e^(σ(s_i - s_j)))) × |ΔNDCG_ij|
```

Where:
- `s_i, s_j` = Scores for items i and j
- `σ` = Sigmoid steepness (typically 1)
- `ΔNDCG_ij` = Change in NDCG if items i and j were swapped

#### Intuition

The lambda gradient says:
> "Move item i up if swapping it with item j would improve NDCG, weighted by how much it would improve"

### ΔNDCG Calculation

```python
def delta_ndcg(y_true, pos_i, pos_j):
    """Change in NDCG if items at pos_i and pos_j are swapped."""
    
    # Gain difference
    gain_i = (2 ** y_true[pos_i] - 1)
    gain_j = (2 ** y_true[pos_j] - 1)
    
    # Discount at each position
    discount_i = 1 / log2(pos_i + 2)
    discount_j = 1 / log2(pos_j + 2)
    
    # Change in DCG
    delta_dcg = (gain_i - gain_j) * (discount_i - discount_j)
    
    return abs(delta_dcg / IDCG)
```

### Gradient Boosting Framework

LambdaMART uses gradient boosting:

```
F_0(x) = initial_prediction  # Usually mean relevance
F_m(x) = F_{m-1}(x) + η × h_m(x)  # Add tree h_m with learning rate η
```

Each tree `h_m` is trained to predict the **lambda gradients**:

```python
# Pseudocode for one boosting iteration
for m in range(num_trees):
    # Compute lambda gradients for all pairs
    lambdas = compute_lambda_gradients(scores, relevances)
    
    # Train tree to predict lambdas
    tree_m = DecisionTree.fit(X, lambdas)
    
    # Update model
    F_m = F_{m-1} + learning_rate * tree_m.predict(X)
```

---

## 3. Why LightGBM/XGBoost vs Other Options

### Comparison of Ranking Algorithms

| Algorithm | Type | Pros | Cons | Use Case |
|-----------|------|------|------|----------|
| **LightGBM LambdaMART** | Listwise GBDT | Fast, handles large data, direct NDCG optimization | Requires feature engineering | Production ranking |
| **XGBoost** | Listwise GBDT | Robust, widely supported | Slower than LightGBM | When stability matters |
| **Neural Rankers** | Deep Learning | Automatic feature learning, can use raw text | Needs lots of data, slower inference | Text-heavy, large scale |
| **BM25** | Sparse Retrieval | Fast, interpretable | No personalization | First-stage retrieval |
| **Matrix Factorization** | Collaborative | Good for sparse interactions | Cold start problem | User-item CF |
| **Linear Models** | Pointwise | Fast, interpretable | Can't capture non-linear patterns | Baseline |

### Why We Chose LightGBM

1. **Speed**: 10-20x faster than XGBoost for large datasets

```
Dataset: 1M examples, 100 features
XGBoost: ~15 minutes
LightGBM: ~45 seconds
```

2. **Memory Efficiency**: Histogram-based splitting

```
XGBoost: O(data × features) for split finding
LightGBM: O(bins × features) where bins << data
```

3. **Native LambdaMART**: Built-in `objective='lambdarank'`

4. **Categorical Support**: Native handling without one-hot encoding

### LightGBM vs XGBoost: Technical Differences

| Aspect | LightGBM | XGBoost |
|--------|----------|---------|
| Tree Growth | Leaf-wise (best-first) | Level-wise (breadth-first) |
| Split Finding | Histogram-based | Exact or histogram |
| Parallelization | Feature + data parallel | Feature parallel |
| Categorical | Native | Requires encoding |
| Memory | Lower | Higher |

#### Leaf-wise vs Level-wise

```
Level-wise (XGBoost):        Leaf-wise (LightGBM):
       [root]                      [root]
      /      \                    /      \
   [L1]      [L1]              [L1]      [L1]
   / \       / \               / \          
[L2][L2]  [L2][L2]          [L2][L2]    (grows best leaf only)
```

Leaf-wise is faster but can overfit - controlled by `max_depth` and `num_leaves`.

### Why Not Neural Networks?

Neural rankers (like TF-Ranking, BERT-based) are powerful but:

1. **Inference latency**: 50-200ms vs 5-10ms for GBDT
2. **Data requirements**: Need millions of examples
3. **Interpretability**: GBDT provides feature importance
4. **Deployment complexity**: GPUs, model serving infrastructure

**When to use Neural Rankers:**
- You have raw text features
- Dataset > 10M examples
- Latency budget > 100ms
- Have GPU infrastructure

---

## 4. Feature Engineering

### Feature Categories

#### 1. Item Features (Static)

```python
# Numeric features
"star_rating"      # 1-5 scale
"review_score"     # 0-5 continuous
"price_per_night"  # Dollar amount

# Computed features
"popularity_score" = booking_count / max_booking_count
"quality_score"    = (review_score × 0.7) + (star_rating/5 × 0.3)
"freshness_score"  = exp(-days_since_update / 30)
```

#### 2. User Features (Personalization)

```python
# Historical behavior
"avg_spend"              # Average booking value
"booking_history_count"  # Total bookings
"preferred_star_rating"  # Mode of past bookings

# Recency
"days_since_last_booking" = (today - last_booking_date).days
```

#### 3. Cross Features (User-Item Interaction)

Cross features capture the **interaction** between user and item:

```python
# Price alignment
"price_vs_avg_spend" = (item_price - user_avg_spend) / user_avg_spend

# Example: User avg=$150, Item=$200
# price_vs_avg_spend = (200-150)/150 = 0.33 (33% above user's average)

# Preference match
"star_preference_match" = 1 - |preferred_star - item_star| / 5

# Example: User prefers 4-star, Item is 3-star
# star_preference_match = 1 - |4-3|/5 = 0.8
```

#### 4. Context Features (Real-time)

```python
# Device
"is_mobile" = 1 if device == "mobile" else 0

# Time
"hour_of_day" = timestamp.hour / 24  # Normalized 0-1
"is_weekend"  = 1 if timestamp.weekday() >= 5 else 0

# Session
"items_viewed_in_session"  # Engagement signal
```

#### 5. GenAI Features (Semantic)

```python
# Embedding similarity
"embedding_similarity" = cosine_similarity(user_embedding, item_embedding)

# Formula:
# cos(u, v) = (u · v) / (||u|| × ||v||)
# where u · v = Σ u_i × v_i (dot product)
# ||u|| = sqrt(Σ u_i²) (L2 norm)

# Summary-based
"summary_length"      # Proxy for content richness
"sentiment_score"     # Extracted from LLM summary
```

### Feature Normalization

```python
# Z-score normalization
normalized = (value - mean) / std

# Min-max normalization
normalized = (value - min) / (max - min)

# Log transformation (for skewed distributions like price)
normalized = log(1 + value)
```

### Feature Importance Analysis

After training, LambdaMART provides feature importance:

```python
# Gain-based importance
importance[feature] = Σ (gain from splits using this feature)

# Split-based importance
importance[feature] = count of splits using this feature
```

---

## 5. Approximate Nearest Neighbors (ANN) vs KNN

### The Problem

Given a query embedding, find the K most similar items from a large catalog.

### Exact KNN

```python
def exact_knn(query, items, k):
    """O(n × d) complexity"""
    distances = []
    for item in items:  # O(n)
        dist = cosine_distance(query, item)  # O(d)
        distances.append(dist)
    return top_k(distances, k)
```

**Time Complexity**: O(n × d) where n = items, d = dimensions

| Items | Dimensions | Time (single query) |
|-------|------------|---------------------|
| 10K | 384 | ~5ms |
| 100K | 384 | ~50ms |
| 1M | 384 | ~500ms |
| 10M | 384 | ~5 seconds |

**KNN doesn't scale!**

### Approximate Nearest Neighbors (ANN)

ANN trades **accuracy for speed** by building index structures.

#### Why "Approximate"?

ANN may not return the exact K nearest neighbors, but returns items that are **close enough** with high probability.

```
Exact KNN:     Returns items ranked 1, 2, 3, 4, 5
ANN (recall=0.95): Might return items ranked 1, 2, 4, 5, 7
```

For recommendations, this is acceptable because:
1. We're retrieving candidates for re-ranking (not final output)
2. Small differences in similarity don't affect user experience
3. 100x speedup is worth 5% recall loss

### FAISS Index Types

CusFlow uses [FAISS](https://faiss.ai/) (Facebook AI Similarity Search):

#### 1. Flat Index (Exact)
```python
index = faiss.IndexFlatIP(dimension)  # Inner Product
# O(n × d) - exact but slow
```

#### 2. IVF (Inverted File Index)
```python
# Cluster vectors into nlist clusters
# Only search nprobe clusters at query time

index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.nprobe = 10  # Search 10 clusters

# Complexity: O(nprobe × n/nlist × d)
# If nlist=100, nprobe=10: ~10x speedup
```

**How IVF Works:**

```
Training:
1. Cluster all vectors into 100 clusters (K-means)
2. Store cluster centroids

Query:
1. Find 10 closest cluster centroids  O(100 × d)
2. Search only vectors in those 10 clusters  O(10 × n/100 × d) = O(n/10 × d)

Total: ~10x faster than brute force
```

#### 3. HNSW (Hierarchical Navigable Small World)

```python
index = faiss.IndexHNSWFlat(dimension, M)
# M = number of connections per node

# Complexity: O(log(n) × d)
```

**How HNSW Works:**

Builds a multi-layer graph where:
- Top layers have few nodes (for fast navigation)
- Bottom layers have all nodes (for accuracy)

```
Layer 2:  A -------- B           (few nodes, long jumps)
          |          |
Layer 1:  A -- C -- B -- D       (more nodes)
          |    |    |    |
Layer 0:  A-C-E-B-D-F-G-H        (all nodes, short jumps)

Query: Start at top layer, greedily descend
```

### ANN Comparison

| Method | Build Time | Query Time | Recall@10 | Memory |
|--------|-----------|------------|-----------|--------|
| Flat (exact) | O(1) | O(n) | 100% | 1x |
| IVF | O(n) | O(n/nlist) | 95-99% | 1x |
| HNSW | O(n log n) | O(log n) | 95-99% | 1.5-2x |
| PQ (Product Quantization) | O(n) | O(n) | 85-95% | 0.1x |

### Why CusFlow Uses ANN

```python
# In candidate_generation/ann_retriever.py

if n_items < 10_000:
    index_type = "Flat"  # Exact search is fast enough
elif n_items < 1_000_000:
    index_type = "IVF"   # Good balance
else:
    index_type = "HNSW"  # Best for very large catalogs
```

**Real-world numbers:**
- 100K items, 384-dim embeddings
- KNN: 50ms per query
- IVF (nprobe=10): 3ms per query
- HNSW: 0.5ms per query

---

## 6. Embeddings & Similarity

### What are Embeddings?

Embeddings are dense vector representations that capture semantic meaning:

```
"luxury hotel with pool" → [0.23, -0.45, 0.12, ..., 0.87]  # 384 dimensions
"budget motel no amenities" → [-0.12, 0.33, -0.56, ..., -0.23]
```

Similar items have similar vectors (close in vector space).

### Sentence Transformers

CusFlow uses `sentence-transformers` for local embeddings:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("luxury 5-star hotel with spa")
# Shape: (384,)
```

**Model Architecture:**
```
Input: "luxury hotel with pool"
   ↓
Tokenize: [101, 6203, 3309, 2007, 4770, 102]
   ↓
BERT-like encoder (6 layers, 384 hidden)
   ↓
Mean pooling over tokens
   ↓
Output: [0.23, -0.45, ..., 0.87]  # 384-dim
```

### Similarity Functions

#### Cosine Similarity (Used in CusFlow)

```python
def cosine_similarity(a, b):
    """
    cos(θ) = (a · b) / (||a|| × ||b||)
    Range: [-1, 1] where 1 = identical direction
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)
```

**Why Cosine?**
- Scale-invariant (magnitude doesn't matter)
- Works well for normalized embeddings
- Common in NLP/semantic similarity

#### Euclidean Distance

```python
def euclidean_distance(a, b):
    """
    ||a - b||₂ = sqrt(Σ (a_i - b_i)²)
    Range: [0, ∞) where 0 = identical
    """
    return np.linalg.norm(a - b)
```

#### Inner Product (Dot Product)

```python
def inner_product(a, b):
    """
    a · b = Σ a_i × b_i
    For normalized vectors: equivalent to cosine similarity
    """
    return np.dot(a, b)
```

**FAISS uses Inner Product** with L2-normalized vectors (equivalent to cosine).

### Using Embeddings as Features

```python
# Embedding similarity as ranking feature
user_embedding = embed_user_preferences(user)
item_embedding = embed_item(item)

similarity = cosine_similarity(user_embedding, item_embedding)
features["embedding_sim"] = similarity  # Input to LambdaMART
```

---

## 7. Click Bias Correction

### The Position Bias Problem

Users click items at higher positions more often, regardless of relevance:

```
Position 1: 30% CTR
Position 2: 20% CTR
Position 3: 12% CTR
Position 4: 8% CTR
...
```

If we train on raw clicks, the model learns:
> "Items that were shown at position 1 are good"

Instead of:
> "Items that users actually prefer are good"

### Click Model

We model clicks as:

```
P(click | item, position) = P(examine | position) × P(click | examine, item)
```

Where:
- `P(examine | position)` = Probability user looks at this position
- `P(click | examine, item)` = True relevance (what we want to learn)

### Position Bias Models

#### Power Law Model (Simple)

```python
P(examine | position) = 1 / position^α

# α typically between 0.5 and 1.5
# Higher α = steeper drop-off

# Example with α = 1:
# Position 1: P(examine) = 1.0
# Position 2: P(examine) = 0.5
# Position 3: P(examine) = 0.33
# Position 5: P(examine) = 0.2
```

#### Exponential Decay Model

```python
P(examine | position) = exp(-λ × (position - 1))

# Example with λ = 0.3:
# Position 1: P(examine) = 1.0
# Position 2: P(examine) = 0.74
# Position 3: P(examine) = 0.55
# Position 5: P(examine) = 0.30
```

### Inverse Propensity Weighting (IPW)

IPW corrects for bias by **reweighting** training examples:

```python
# For each clicked item:
weight = 1 / P(examine | position)

# If item at position 3 was clicked:
# weight = 1 / 0.33 = 3.0

# Intuition: A click at position 3 is "worth more" than a click at position 1
# because it's harder to get clicks at lower positions
```

**Training with IPW:**

```python
# Standard training
model.fit(X, y)  # All examples weighted equally

# IPW training
weights = compute_ipw_weights(positions, clicks)
model.fit(X, y, sample_weight=weights)
```

### Mathematical Derivation

The unbiased estimator for relevance is:

```
E[relevance] = E[click / P(examine | position)]

Proof:
E[click / P(examine)] 
= E[P(examine) × P(click|examine) / P(examine)]
= E[P(click | examine)]
= true_relevance
```

### Doubly Robust Estimation

Combines IPW with a relevance model for lower variance:

```python
# Doubly robust estimator
DR = relevance_model(item) + (click - P(examine) × relevance_model(item)) / P(examine)

# If either the propensity model OR the relevance model is correct,
# the estimator is unbiased
```

---

## 8. Evaluation Metrics

### NDCG (Normalized Discounted Cumulative Gain)

The primary metric for ranking quality.

#### DCG (Discounted Cumulative Gain)

```
DCG@K = Σᵢ₌₁ᵏ (2^relᵢ - 1) / log₂(i + 1)

Where:
- relᵢ = relevance of item at position i (typically 0-4)
- log₂(i + 1) = discount factor (position penalty)
```

**Example:**
```
Ranking: [rel=3, rel=1, rel=2, rel=0]

DCG@4 = (2³-1)/log₂(2) + (2¹-1)/log₂(3) + (2²-1)/log₂(4) + (2⁰-1)/log₂(5)
      = 7/1 + 1/1.58 + 3/2 + 0/2.32
      = 7 + 0.63 + 1.5 + 0
      = 9.13
```

#### IDCG (Ideal DCG)

DCG of the perfect ranking (sorted by relevance):

```
Perfect ranking: [rel=3, rel=2, rel=1, rel=0]

IDCG@4 = (2³-1)/1 + (2²-1)/1.58 + (2¹-1)/2 + (2⁰-1)/2.32
       = 7 + 1.90 + 0.5 + 0
       = 9.40
```

#### NDCG

```
NDCG@K = DCG@K / IDCG@K

NDCG@4 = 9.13 / 9.40 = 0.97
```

**Interpretation:**
- NDCG = 1.0: Perfect ranking
- NDCG = 0.5: Mediocre ranking
- NDCG = 0.0: All relevant items at bottom

### MAP (Mean Average Precision)

#### Precision@K

```
Precision@K = (# relevant items in top K) / K

Example: Top 5 results, 3 are relevant
Precision@5 = 3/5 = 0.6
```

#### Average Precision (AP)

```
AP = Σₖ (Precision@k × rel(k)) / (# relevant items)

Where rel(k) = 1 if item at position k is relevant, else 0
```

**Example:**
```
Ranking: [R, N, R, N, R]  (R=relevant, N=not relevant)

P@1 = 1/1 = 1.0  (R)   → contributes 1.0 × 1
P@2 = 1/2 = 0.5  (N)   → contributes 0
P@3 = 2/3 = 0.67 (R)   → contributes 0.67 × 1
P@4 = 2/4 = 0.5  (N)   → contributes 0
P@5 = 3/5 = 0.6  (R)   → contributes 0.6 × 1

AP = (1.0 + 0.67 + 0.6) / 3 = 0.76
```

#### MAP

```
MAP = (1/Q) × Σ AP_q  for all queries q
```

### Recall@K

```
Recall@K = (# relevant items in top K) / (total # relevant items)

Example: 10 relevant items total, 6 in top 20
Recall@20 = 6/10 = 0.6
```

### MRR (Mean Reciprocal Rank)

```
RR = 1 / (position of first relevant item)

MRR = (1/Q) × Σ RR_q

Example:
Query 1: First relevant at position 3 → RR = 1/3
Query 2: First relevant at position 1 → RR = 1/1
Query 3: First relevant at position 5 → RR = 1/5

MRR = (1/3 + 1 + 1/5) / 3 = 0.51
```

### Metric Summary

| Metric | Focuses On | Range | Best For |
|--------|-----------|-------|----------|
| **NDCG@K** | Graded relevance, position | [0, 1] | Multi-level relevance |
| **MAP** | Binary relevance, full ranking | [0, 1] | Binary judgments |
| **Recall@K** | Coverage | [0, 1] | Ensuring relevant items appear |
| **MRR** | First relevant item | [0, 1] | Navigational queries |
| **Precision@K** | Top-K quality | [0, 1] | Short lists |

---

## Summary: Why These Choices?

| Choice | Reason |
|--------|--------|
| **LambdaMART** | Direct NDCG optimization, fast inference, interpretable |
| **LightGBM** | Fastest GBDT implementation, handles large data |
| **ANN (not KNN)** | O(log n) vs O(n) query time at scale |
| **FAISS IVF/HNSW** | Production-proven, tunable accuracy/speed tradeoff |
| **Cosine Similarity** | Scale-invariant, works well with text embeddings |
| **IPW for Bias** | Theoretically sound, simple to implement |
| **NDCG as Primary Metric** | Handles graded relevance, penalizes bad top results |

---

## References

1. Burges, C. (2010). "From RankNet to LambdaRank to LambdaMART"
2. Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
3. Johnson, J. et al. (2019). "Billion-scale similarity search with GPUs" (FAISS)
4. Joachims, T. et al. (2017). "Unbiased Learning-to-Rank with Biased Feedback"
5. Wang, X. et al. (2018). "Position Bias Estimation for Unbiased Learning to Rank"
