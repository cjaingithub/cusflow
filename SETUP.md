# CusFlow Setup Guide

Complete guide to get CusFlow running on your machine.

## Quick Start (5 minutes)

### Prerequisites

- Python 3.10+ installed
- Git installed
- (Optional) Docker for Redis

### Step 1: Clone & Install

```bash
# Clone the repository
git clone https://github.com/cjaingithub/cusflow.git
cd cusflow

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### Step 2: Verify Installation

```bash
# Check installation
python -c "from src.config import get_settings; print('✓ CusFlow installed successfully!')"

# Run tests
pytest tests/test_ranking.py -v
```

### Step 3: Generate Sample Data

```bash
# Generate hotel recommendation data
python -m src.cli generate-data --domain hotel --items 500 --users 200

# Verify data was created
ls data/
# Should show: items.parquet, users.parquet, events.parquet, training.parquet
```

### Step 4: Train a Model

```bash
# Train LambdaMART ranking model
python scripts/train_model.py --estimators 100

# Verify model was saved
ls models/
# Should show: lambdamart_v1.joblib, lambdamart_v1.json, lambdamart_v1.lgb
```

### Step 5: Start the API (Optional)

```bash
# Start Redis (required for API)
docker run -d -p 6379:6379 --name cusflow-redis redis:7-alpine

# Load features into Redis
python scripts/load_features.py

# Start the API server
python -m src.cli serve --port 8000

# Test the API (in another terminal)
curl http://localhost:8000/health
```

## Detailed Setup

### Installing Dependencies

#### Production Only

```bash
pip install -e .
```

#### With Development Tools

```bash
pip install -e ".[dev]"
```

#### Using requirements.txt

```bash
pip install -r requirements.txt
```

### Environment Configuration

```bash
# Copy example configuration
cp env.example .env

# Edit .env file (optional - defaults work for local testing)
```

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `DOMAIN` | `hotel` | Business domain (hotel, wealth_report, ecommerce) |
| `REDIS_HOST` | `localhost` | Redis server host |
| `REDIS_PORT` | `6379` | Redis server port |
| `USE_LOCAL_EMBEDDINGS` | `true` | Use local sentence-transformers |
| `OPENAI_API_KEY` | (empty) | Optional: For OpenAI embeddings |

### Redis Setup Options

#### Option 1: Docker (Recommended)

```bash
docker run -d -p 6379:6379 --name cusflow-redis redis:7-alpine
```

#### Option 2: Docker Compose

```bash
docker compose up -d redis
```

#### Option 3: Local Installation

- **Windows**: Download from https://github.com/microsoftarchive/redis/releases
- **macOS**: `brew install redis && brew services start redis`
- **Linux**: `sudo apt install redis-server && sudo systemctl start redis`

### Testing Your Setup

#### 1. Run Unit Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

#### 2. Test Data Generation

```bash
# Generate data for different domains
python -m src.cli generate-data --domain hotel --items 100
python -m src.cli generate-data --domain wealth_report --items 100
python -m src.cli generate-data --domain ecommerce --items 100
```

#### 3. Test Model Training

```bash
# Quick training test (fewer iterations)
python scripts/train_model.py --estimators 10

# Full training
python scripts/train_model.py --estimators 500
```

#### 4. Test API

```bash
# Start API
python -m src.cli serve --port 8000

# In another terminal:
python scripts/demo_api.py
```

#### 5. Test Evaluation

```bash
# Run evaluation
python scripts/evaluate.py

# With ablation study
python scripts/evaluate.py --ablation
```

#### 6. Test A/B Simulation

```bash
python scripts/run_ab_sim.py --generate
```

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'src'"

```bash
# Make sure you're in the project directory and installed with -e
cd cusflow
pip install -e .
```

#### 2. "Redis connection refused"

```bash
# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Or check if already running
docker ps | grep redis
```

#### 3. "No module named 'lightgbm'"

```bash
pip install lightgbm>=4.3.0
```

#### 4. Tests failing with import errors

```bash
# Reinstall with dev dependencies
pip install -e ".[dev]"
```

#### 5. FAISS installation issues on Windows

```bash
# Use CPU version
pip install faiss-cpu
```

### Getting Help

1. Check existing [GitHub Issues](https://github.com/cjaingithub/cusflow/issues)
2. Open a new issue with:
   - Your OS and Python version
   - Full error message
   - Steps to reproduce

## Next Steps

After setup, try:

1. **Explore the notebook**: `jupyter notebook notebooks/exploration.ipynb`
2. **Read the docs**: See `docs/` folder
3. **Customize for your domain**: Edit `src/config.py`
4. **Contribute**: See `CONTRIBUTING.md`

## Uninstalling

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv/

# Remove data and models
rm -rf data/ models/ reports/
```
