# CusFlow Architecture

## System Overview

CusFlow is a modular recommendation system designed for production deployment. The architecture follows a microservices pattern with clear separation of concerns.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 API Layer                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         FastAPI Application                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │    │
│  │  │   /rank     │  │   /items    │  │   /users    │  │  /health   │ │    │
│  │  │   Endpoint  │  │   Endpoint  │  │   Endpoint  │  │  Endpoint  │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Service Layer                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         Ranking Pipeline                             │    │
│  │                                                                      │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │    │
│  │  │  Candidate  │───▶│   Feature   │───▶│   LTR       │             │    │
│  │  │  Generation │    │  Engineering │    │   Scoring   │             │    │
│  │  └─────────────┘    └─────────────┘    └─────────────┘             │    │
│  │        │                   │                  │                     │    │
│  │        ▼                   ▼                  ▼                     │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │    │
│  │  │   FAISS     │    │   GenAI     │    │ LambdaMART  │             │    │
│  │  │   Index     │    │  Features   │    │   Model     │             │    │
│  │  └─────────────┘    └─────────────┘    └─────────────┘             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Storage Layer                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │     Redis       │  │   File System   │  │    External     │             │
│  │  Feature Store  │  │  (Models/Data)  │  │   APIs (LLM)    │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Ranking Request Flow

```
User Request
     │
     ▼
┌────────────┐
│  FastAPI   │──────────────────────────────────────┐
│  Router    │                                      │
└────────────┘                                      │
     │                                              │
     ▼                                              │
┌────────────┐    ┌────────────┐                   │
│   Get      │───▶│   Redis    │  User/Item        │
│   Features │    │   Store    │  Features         │
└────────────┘    └────────────┘                   │
     │                                              │
     ▼                                              │
┌────────────┐    ┌────────────┐                   │
│  Feature   │───▶│  Build     │  Feature          │
│  Engineer  │    │  Matrix    │  Vector           │
└────────────┘    └────────────┘                   │
     │                                              │
     ▼                                              │
┌────────────┐    ┌────────────┐                   │
│ LambdaMART │───▶│  Predict   │  Ranking          │
│   Model    │    │  Scores    │  Scores           │
└────────────┘    └────────────┘                   │
     │                                              │
     ▼                                              │
┌────────────┐                                      │
│   Return   │◀─────────────────────────────────────┘
│  Response  │
└────────────┘
```

### 2. Training Pipeline Flow

```
Raw Data (Events, Items, Users)
     │
     ▼
┌────────────┐
│   Data     │
│   Loader   │
└────────────┘
     │
     ├─────────────────────────────┐
     ▼                             ▼
┌────────────┐              ┌────────────┐
│  Feature   │              │   GenAI    │
│  Extract   │              │  Embed     │
└────────────┘              └────────────┘
     │                             │
     └──────────┬──────────────────┘
                ▼
         ┌────────────┐
         │  Training  │
         │   Data     │
         └────────────┘
                │
                ▼
         ┌────────────┐
         │   Bias     │  (Optional)
         │ Correction │
         └────────────┘
                │
                ▼
         ┌────────────┐
         │ LambdaMART │
         │   Train    │
         └────────────┘
                │
                ▼
         ┌────────────┐
         │   Save     │
         │   Model    │
         └────────────┘
```

## Key Design Decisions

### 1. Domain-Agnostic Design

The system uses configuration-driven feature definitions:

```python
DomainConfig.CONFIGS = {
    Domain.HOTEL: {
        "features": ["star_rating", "review_score", ...],
        "summary_prompt": "...",
    },
    Domain.WEALTH_REPORT: {
        "features": ["risk_level", "return_potential", ...],
        "summary_prompt": "...",
    },
}
```

### 2. Feature Store Pattern

Redis serves as a lightweight feature store:
- **Item features**: Cached with TTL
- **User features**: Real-time updates
- **Embeddings**: Pre-computed vectors

### 3. Two-Stage Ranking

1. **Candidate Generation**: Fast ANN retrieval (FAISS)
2. **Re-ranking**: LambdaMART for final ordering

### 4. Bias Correction

Inverse Propensity Weighting (IPW) corrects for position bias:

```
weight_i = 1 / P(examine | position_i)
```

## Scalability Considerations

| Component | Scaling Strategy |
|-----------|-----------------|
| API | Horizontal scaling with load balancer |
| Redis | Redis Cluster or managed service |
| Model | Model sharding by user segment |
| Embeddings | GPU inference, batching |

## Security

- Environment-based configuration
- No secrets in code
- Redis authentication support
- CORS configuration for API
