"""
CusFlow - Production-Ready Learning-to-Rank Recommendation System

A domain-agnostic recommendation platform combining:
- Learning-to-Rank (LambdaMART) for personalized ranking
- GenAI features (LLM summaries + embeddings) for semantic understanding
- A/B simulation for uplift estimation
- Production deployment with FastAPI, Redis, and Docker
"""

__version__ = "1.0.0"
__author__ = "CusFlow Team"

from src.config import Domain, Settings, get_settings

__all__ = ["Domain", "Settings", "get_settings", "__version__"]
