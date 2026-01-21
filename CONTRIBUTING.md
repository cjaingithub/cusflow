# Contributing to CusFlow

Thank you for your interest in contributing to CusFlow! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Docker (optional, for running Redis)
- 4GB+ RAM recommended for training models

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/cusflow.git
cd cusflow
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/cjaingithub/cusflow.git
```

## Development Setup

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or using requirements.txt
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov ruff mypy pre-commit
```

### 3. Set Up Pre-commit Hooks

```bash
pre-commit install
```

### 4. Configure Environment

```bash
# Copy example env file
cp env.example .env

# Edit .env with your settings (optional for basic testing)
```

### 5. Start Redis (Optional)

For full functionality, you'll need Redis:

```bash
# Using Docker
docker run -d -p 6379:6379 --name cusflow-redis redis:7-alpine

# Or use Docker Compose
docker compose up -d redis
```

## Running Tests

### Quick Test

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_ranking.py -v
```

### Full Test Suite

```bash
# Run linting + type checking + tests
make lint
make test
```

### Testing Without Redis

Most tests work without Redis. For full integration tests:

```bash
# Start Redis first
docker run -d -p 6379:6379 redis:7-alpine

# Run tests
pytest tests/ -v
```

### Manual Testing

```bash
# 1. Generate test data
python -m src.cli generate-data --domain hotel --items 100 --users 50

# 2. Train a model
python scripts/train_model.py --estimators 50

# 3. Start the API
python -m src.cli serve --port 8000

# 4. Test the API (in another terminal)
curl http://localhost:8000/health
python scripts/demo_api.py
```

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

### Style Guidelines

- **Line length**: 100 characters max
- **Imports**: Sorted with isort (via Ruff)
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public modules, classes, and functions

### Running Linters

```bash
# Check code style
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/

# Type checking
mypy src/ --ignore-missing-imports
```

### Example Code Style

```python
"""Module docstring explaining the purpose."""

from typing import Any

import numpy as np

from src.config import get_settings


def calculate_score(
    features: np.ndarray,
    weights: list[float],
    normalize: bool = True,
) -> float:
    """
    Calculate weighted score from features.

    Args:
        features: Feature vector of shape (n_features,)
        weights: Weight for each feature
        normalize: Whether to normalize the result

    Returns:
        Calculated score value

    Raises:
        ValueError: If features and weights have different lengths
    """
    if len(features) != len(weights):
        raise ValueError("Features and weights must have same length")

    score = np.dot(features, weights)

    if normalize:
        score = score / (np.sum(weights) + 1e-8)

    return float(score)
```

## Submitting Changes

### 1. Create a Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, documented code
- Add tests for new functionality
- Update documentation if needed

### 3. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new ranking algorithm

- Implemented BM25 ranking
- Added unit tests
- Updated documentation"
```

#### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `style:` Formatting changes
- `chore:` Maintenance tasks

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

### Pull Request Guidelines

- Fill out the PR template completely
- Link related issues
- Ensure all tests pass
- Request review from maintainers

## Reporting Issues

### Bug Reports

Include:
1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal steps to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: OS, Python version, package versions
6. **Logs/Screenshots**: Any relevant output

### Feature Requests

Include:
1. **Problem**: What problem does this solve?
2. **Proposed Solution**: How would you implement it?
3. **Alternatives**: Other solutions you considered
4. **Use Case**: Who would benefit from this?

## Project Structure

```
cusflow/
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── candidate_generation/  # ANN retrieval
│   ├── data/             # Data schemas & loaders
│   ├── evaluation/       # Metrics & A/B testing
│   ├── genai/            # LLM & embeddings
│   ├── ranking/          # LTR models
│   ├── store/            # Redis feature store
│   ├── cli.py            # CLI commands
│   └── config.py         # Configuration
├── tests/                # Test suite
├── scripts/              # Utility scripts
├── notebooks/            # Jupyter notebooks
├── docs/                 # Documentation
└── docker-compose.yml    # Docker setup
```

## Areas for Contribution

### Good First Issues

- Adding more unit tests
- Improving documentation
- Adding type hints to existing code
- Fixing linting warnings

### Feature Ideas

- Additional ranking algorithms (BM25, neural rankers)
- More embedding providers (Cohere, local LLMs)
- Dashboard for metrics visualization
- Batch inference optimization
- Additional domain configurations

## Getting Help

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check `docs/` folder

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to CusFlow! 🚀
