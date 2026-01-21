# CusFlow Quick Reference

## CLI Commands

```bash
# Generate data
cusflow generate-data --domain hotel --items 1000 --users 500

# Start server
cusflow serve --port 8000 --reload

# Show configuration
cusflow info
```

## Make Commands

```bash
make install        # Install dependencies
make dev            # Install with dev tools
make generate-data  # Generate hotel data
make train          # Train model
make evaluate       # Evaluate model
make ab-sim         # Run A/B simulation
make serve          # Start API
make docker-up      # Start Docker stack
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/rank` | POST | Get rankings |
| `/items` | POST | Add items |
| `/items/{id}` | GET | Get item |
| `/users` | POST | Add user |
| `/embeddings/items` | POST | Generate embeddings |

## Environment Variables

```bash
# Required
DOMAIN=hotel
REDIS_HOST=localhost

# Optional
OPENAI_API_KEY=sk-...
USE_LOCAL_EMBEDDINGS=true
```

## Python API

```python
# Config
from src.config import get_settings
settings = get_settings()

# Data
from src.data.loaders import SyntheticDataGenerator
gen = SyntheticDataGenerator(domain=Domain.HOTEL)
items = gen.generate_items(1000)

# Ranking
from src.ranking.lambdamart import LambdaMARTRanker
model = LambdaMARTRanker()
model.fit(X, y, groups)
scores = model.predict(X_test)

# Evaluation
from src.evaluation.metrics import RankingMetrics
metrics = RankingMetrics()
results = metrics.evaluate(y_true, y_pred, groups)
```

## Docker

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f api

# Run with dev profile
docker compose --profile dev up -d

# Stop
docker compose down
```

## File Structure

```
src/
├── api/        # FastAPI app
├── data/       # Schemas, loaders
├── ranking/    # LambdaMART, features
├── genai/      # Embeddings, summaries
├── evaluation/ # Metrics, A/B sim
├── store/      # Redis
└── config.py   # Settings
```

## Common Patterns

### Load and rank
```python
from src.data.loaders import DataLoader
from src.ranking.lambdamart import LambdaMARTRanker

loader = DataLoader("data/")
X, y, groups = loader.load_training_data()

model = LambdaMARTRanker()
model.fit(X, y, groups)
model.save("models/model.joblib")
```

### Evaluate with ablation
```python
from src.evaluation.ablation import AblationStudy

study = AblationStudy(model_class=LambdaMARTRanker)
result = study.run(X, y, groups, feature_names)
print(study.generate_report(result))
```

### Run A/B simulation
```python
from src.evaluation.ab_simulation import ABSimulator

simulator = ABSimulator(control, treatment)
result = simulator.simulate(events, users, items)
print(simulator.generate_report(result))
```
