# 🚀 CusFlow

**Production-Ready Learning-to-Rank Recommendation System with GenAI Features**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CusFlow is a domain-agnostic recommendation platform that combines **Learning-to-Rank (LambdaMART)** with **GenAI features** (LLM summaries + embeddings) to deliver personalized rankings. Built for production with FastAPI, Redis, and Docker.

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **🎓 LambdaMART Ranking** | Train LightGBM-based LTR models with click bias correction |
| **🤖 GenAI Integration** | LLM item summaries + sentence-transformer embeddings |
| **📊 Offline Evaluation** | NDCG@K, MAP, Recall@K, MRR with ablation studies |
| **🧪 A/B Simulation** | Replay historical traffic to estimate CTR/CVR uplift |
| **⚡ Real-time Serving** | FastAPI + Redis feature store for low-latency inference |
| **🐳 Production Ready** | Fully Dockerized with Metarank integration |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CusFlow Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Client     │───▶│   FastAPI    │───▶│   Ranking    │                   │
│  │   Request    │    │   Gateway    │    │   Service    │                   │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                   │
│                                                  │                           │
│         ┌────────────────────────────────────────┼────────────────────┐      │
│         │                                        │                    │      │
│         ▼                                        ▼                    ▼      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   Candidate  │    │    Feature   │    │  LambdaMART  │    │   GenAI    │ │
│  │  Generation  │    │    Store     │    │    Model     │    │  Features  │ │
│  │   (FAISS)    │    │   (Redis)    │    │  (LightGBM)  │    │  (OpenAI)  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Evaluation & Experimentation                  │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │   NDCG@K    │  │    MAP      │  │  A/B Sim    │  │  Ablation   │ │    │
│  │  │  Recall@K   │  │    MRR      │  │   Uplift    │  │   Study     │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/cjaingithub/cusflow.git
cd cusflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Generate Synthetic Data

```bash
# Generate hotel data (default)
python -m src.cli generate-data --domain hotel --items 1000 --users 500

# Or wealth management reports
python -m src.cli generate-data --domain wealth_report --items 500 --users 200

# Or e-commerce products
python -m src.cli generate-data --domain ecommerce --items 2000 --users 1000
```

### 3. Train the Model

```bash
# Train LambdaMART ranking model
python scripts/train_model.py --estimators 500

# With click bias correction
python scripts/train_model.py --estimators 500 --bias-correction
```

### 4. Start Services

```bash
# Start Redis (using Docker)
docker run -d -p 6379:6379 redis:7-alpine

# Load features into Redis
python scripts/load_features.py --embeddings

# Start API server
python -m src.cli serve
```

### 5. Test the API

```bash
# Run demo script
python scripts/demo_api.py

# Or use curl
curl http://localhost:8000/health
```

---

## 🐳 Docker Deployment

### Full Stack Deployment

```bash
# Build and start all services
docker compose up -d

# View logs
docker compose logs -f api

# Stop services
docker compose down
```

### Development Mode

```bash
# Start with hot-reload
docker compose --profile dev up -d dev

# Start Jupyter notebook
docker compose --profile dev up -d jupyter
```

---

## 📊 Supported Domains

CusFlow is designed to be **domain-agnostic**. It works across different recommendation use cases:

### 🏨 Hotels (Expedia-style)

```python
from src.config import Domain
from src.data.loaders import SyntheticDataGenerator

generator = SyntheticDataGenerator(domain=Domain.HOTEL)
items = generator.generate_items(n_items=1000)
# Features: star_rating, review_score, price_per_night, location_score, etc.
```

### 📈 Wealth Management Reports

```python
generator = SyntheticDataGenerator(domain=Domain.WEALTH_REPORT)
items = generator.generate_items(n_items=500)
# Features: asset_class, risk_level, return_potential, author_reputation, etc.
```

### 🛍️ E-commerce Products

```python
generator = SyntheticDataGenerator(domain=Domain.ECOMMERCE)
items = generator.generate_items(n_items=2000)
# Features: price, discount, rating, review_count, stock_status, etc.
```

---

## 🎓 Core Components

### 1. LambdaMART Ranking

```python
from src.ranking.lambdamart import LambdaMARTRanker

# Train model
model = LambdaMARTRanker(num_boost_round=500)
model.fit(X_train, y_train, groups_train,
          X_val=X_val, y_val=y_val, groups_val=groups_val)

# Get feature importance
importance = model.get_feature_importance(top_k=10)

# Rank items
ranked = model.rank(X_test, item_ids, top_k=20)
```

### 2. GenAI Features

```python
from src.genai.embeddings import EmbeddingService
from src.genai.summarizer import ItemSummarizer

# Generate embeddings
embedding_service = EmbeddingService()
embeddings = embedding_service.embed_items(items)

# Generate LLM summaries
summarizer = ItemSummarizer(provider="openai")
summaries = summarizer.summarize_items(items)
```

### 3. Click Bias Correction

```python
from src.ranking.bias_correction import InversePropensityWeighting

# Estimate position bias
ipw = InversePropensityWeighting()
ipw.fit(positions, clicks)

# Get corrected weights for training
weights = ipw.compute_weights(positions, clicks)
model.fit(X, y, groups, sample_weight=weights)
```

### 4. Evaluation

```python
from src.evaluation.metrics import RankingMetrics
from src.evaluation.ablation import AblationStudy

# Evaluate model
metrics = RankingMetrics(cutoffs=[5, 10, 20])
results = metrics.evaluate(y_true, y_pred, groups)
# {'ndcg@10': 0.85, 'map': 0.78, 'recall@10': 0.72, ...}

# Ablation study
study = AblationStudy(model_class=LambdaMARTRanker)
ablation_result = study.run(X, y, groups, feature_names)
print(study.generate_report(ablation_result))
```

### 5. A/B Simulation

```python
from src.evaluation.ab_simulation import ABSimulator, SimulationConfig

config = SimulationConfig(
    experiment_id="genai_features_test",
    treatment_ratio=0.5,
)

simulator = ABSimulator(control_ranker, treatment_ranker, config)
result = simulator.simulate(events, users, items)

print(simulator.generate_report(result))
# Shows CTR/CVR uplift with statistical significance
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/rank` | POST | Get personalized rankings |
| `/items` | POST | Add/update items |
| `/items/{id}` | GET | Get item details |
| `/users` | POST | Add/update users |
| `/embeddings/items` | POST | Generate embeddings |
| `/admin/stats` | GET | Feature store stats |
| `/admin/reload-model` | POST | Reload ranking model |

### Example: Get Rankings

```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "candidate_ids": ["hotel_001", "hotel_002", "hotel_003"],
    "num_results": 10,
    "use_genai_features": true
  }'
```

---

## 📁 Project Structure

```
cusflow/
├── src/
│   ├── api/              # FastAPI application
│   ├── candidate_generation/  # ANN retrieval (FAISS)
│   ├── data/             # Data schemas & loaders
│   ├── evaluation/       # Metrics & A/B simulation
│   ├── genai/            # LLM summaries & embeddings
│   ├── ranking/          # LambdaMART & feature engineering
│   ├── store/            # Redis feature store
│   ├── cli.py            # Command-line interface
│   └── config.py         # Configuration management
├── scripts/
│   ├── train_model.py    # Model training
│   ├── evaluate.py       # Evaluation & ablation
│   ├── run_ab_sim.py     # A/B simulation
│   └── load_features.py  # Load features to Redis
├── tests/                # Test suite
├── notebooks/            # Jupyter notebooks
├── data/                 # Data files (gitignored)
├── models/               # Trained models (gitignored)
├── metarank/             # Metarank configuration
├── docker-compose.yml    # Docker orchestration
├── Dockerfile            # Container definition
└── Makefile              # Task automation
```

---

## 🧪 Running Tests

### Quick Verification

```bash
# Verify installation
python -c "from src.config import get_settings; print('✓ CusFlow installed!')"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Full Test Pipeline

```bash
# 1. Generate test data
python -m src.cli generate-data --domain hotel --items 100 --users 50

# 2. Train a model (quick)
python scripts/train_model.py --estimators 50

# 3. Evaluate
python scripts/evaluate.py

# 4. Run A/B simulation
python scripts/run_ab_sim.py --generate

# 5. Test the API (requires Redis)
docker run -d -p 6379:6379 redis:7-alpine
python scripts/load_features.py
python -m src.cli serve &
python scripts/demo_api.py
```

### Test Without Dependencies

Most core functionality works without Redis or external APIs:

```bash
# These tests work standalone:
pytest tests/test_ranking.py -v
```

---

## 🎤 Why This POC Works for Interviews

This project demonstrates:

| Skill | Evidence |
|-------|----------|
| **ML Engineering** | LambdaMART, feature engineering, bias correction |
| **GenAI Integration** | LLM summaries, embeddings, similarity features |
| **Production Systems** | FastAPI, Redis, Docker, async programming |
| **Experimentation** | A/B simulation, statistical significance, ablation |
| **Software Quality** | Type hints, tests, documentation, CI/CD ready |

### Talking Points

> "I built a ranking platform that integrates LLM features to improve recommendations."

> "The system evaluates offline and simulates online metrics for conversion lift."

> "It's domain-agnostic - works for hotels, financial products, or e-commerce."

---

## 🛠️ Configuration

Copy `.env.example` to `.env` and configure:

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# GenAI (optional)
OPENAI_API_KEY=sk-your-key
USE_LOCAL_EMBEDDINGS=true  # Use free local embeddings

# Domain
DOMAIN=hotel  # hotel, wealth_report, ecommerce
```

---

## 📚 Documentation

- [Architecture Guide](docs/architecture.md)
- [Technical Guide](docs/technical_guide.md)
- [Quick Reference](docs/quick_reference.md)
- [Hotel Recommendation Guide](docs/hotel_recommendation_guide.md)

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [LightGBM](https://lightgbm.readthedocs.io/) for LambdaMART implementation
- [FAISS](https://faiss.ai/) for vector similarity search
- [Metarank](https://metarank.ai/) for ranking engine inspiration
- [sentence-transformers](https://www.sbert.net/) for local embeddings

---

<p align="center">
  Built with ❤️ for the ML/RecSys community
</p>
