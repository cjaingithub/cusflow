"""
CusFlow Configuration Module

Centralized configuration using Pydantic Settings for type-safe environment management.
Supports multiple domains: hotels, wealth reports, e-commerce.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Domain(str, Enum):
    """Supported business domains for the recommendation system."""
    HOTEL = "hotel"
    WEALTH_REPORT = "wealth_report"
    ECOMMERCE = "ecommerce"


class EmbeddingProvider(str, Enum):
    """Available embedding providers."""
    OPENAI = "openai"
    LOCAL = "local"  # sentence-transformers
    ANTHROPIC = "anthropic"


class LLMProvider(str, Enum):
    """Available LLM providers for item summarization."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """Main application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ==========================================================================
    # API Configuration
    # ==========================================================================
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_debug: bool = Field(default=False, description="Enable debug mode")
    
    # ==========================================================================
    # Redis Configuration
    # ==========================================================================
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: str = Field(default="", description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_ttl_seconds: int = Field(default=86400, description="Default TTL for cached features")
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL from components."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # ==========================================================================
    # Metarank Configuration
    # ==========================================================================
    metarank_host: str = Field(default="localhost", description="Metarank host")
    metarank_port: int = Field(default=8080, description="Metarank port")
    metarank_api_key: str = Field(default="", description="Metarank API key")
    
    @property
    def metarank_url(self) -> str:
        """Construct Metarank URL."""
        return f"http://{self.metarank_host}:{self.metarank_port}"
    
    # ==========================================================================
    # GenAI Configuration
    # ==========================================================================
    # OpenAI
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model for summaries")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", 
        description="OpenAI embedding model"
    )
    
    # Anthropic
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022", 
        description="Anthropic model for summaries"
    )
    
    # Local embeddings
    local_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Local sentence-transformer model"
    )
    use_local_embeddings: bool = Field(
        default=True, 
        description="Use local embeddings instead of API"
    )
    
    # Provider selection
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.LOCAL,
        description="Which embedding provider to use"
    )
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Which LLM provider to use for summaries"
    )
    
    # ==========================================================================
    # Model Configuration
    # ==========================================================================
    model_path: Path = Field(default=Path("models/"), description="Path to model files")
    ranking_model_name: str = Field(
        default="lambdamart_v1.joblib",
        description="Ranking model filename"
    )
    candidate_model_name: str = Field(
        default="ann_index.faiss",
        description="ANN index filename"
    )
    
    @field_validator("model_path", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        return Path(v)
    
    @property
    def ranking_model_path(self) -> Path:
        """Full path to ranking model."""
        return self.model_path / self.ranking_model_name
    
    @property
    def candidate_model_path(self) -> Path:
        """Full path to candidate generation model."""
        return self.model_path / self.candidate_model_name
    
    # ==========================================================================
    # Data Configuration
    # ==========================================================================
    data_path: Path = Field(default=Path("data/"), description="Path to data files")
    domain: Domain = Field(default=Domain.HOTEL, description="Business domain")
    
    @field_validator("data_path", mode="before")
    @classmethod
    def ensure_data_path(cls, v: str | Path) -> Path:
        return Path(v)
    
    # ==========================================================================
    # Ranking Configuration
    # ==========================================================================
    ranking_top_k: int = Field(default=100, description="Number of candidates to retrieve")
    ranking_rerank_k: int = Field(default=20, description="Number of items to return after reranking")
    click_bias_correction: bool = Field(default=True, description="Apply click position bias correction")
    
    # ==========================================================================
    # Evaluation Configuration
    # ==========================================================================
    eval_metrics: list[str] = Field(
        default=["ndcg@10", "map", "recall@10", "mrr"],
        description="Evaluation metrics to compute"
    )
    ab_simulation_traffic_ratio: float = Field(
        default=0.5,
        description="Ratio of traffic for treatment in A/B simulation"
    )
    
    @field_validator("eval_metrics", mode="before")
    @classmethod
    def parse_metrics(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [m.strip() for m in v.split(",")]
        return v
    
    # ==========================================================================
    # Logging Configuration
    # ==========================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Log output format"
    )


class DomainConfig:
    """Domain-specific configuration for different use cases."""
    
    CONFIGS = {
        Domain.HOTEL: {
            "item_name": "hotel",
            "item_id_field": "hotel_id",
            "features": [
                "star_rating", "review_score", "price_per_night", "location_score",
                "amenities_count", "distance_to_center", "booking_count", "cancellation_policy"
            ],
            "context_features": [
                "check_in_date", "check_out_date", "guests", "room_type",
                "search_filters", "user_location", "device_type"
            ],
            "user_features": [
                "user_id", "booking_history_count", "avg_spend", "preferred_star_rating",
                "loyalty_tier", "days_since_last_booking"
            ],
            "text_fields": ["name", "description", "amenities_text", "location_description"],
            "summary_prompt": """Summarize this hotel for a traveler in 2-3 sentences. 
Focus on: location highlights, standout amenities, ideal guest type, and value proposition.
Hotel: {item_text}""",
        },
        Domain.WEALTH_REPORT: {
            "item_name": "report",
            "item_id_field": "report_id",
            "features": [
                "asset_class", "risk_level", "return_potential", "time_horizon",
                "min_investment", "publication_date_recency", "author_reputation_score",
                "download_count", "citation_count"
            ],
            "context_features": [
                "investor_profile", "investment_goal", "risk_tolerance",
                "portfolio_size", "current_holdings", "market_conditions"
            ],
            "user_features": [
                "user_id", "aum", "investment_experience_years", "preferred_asset_classes",
                "read_history_count", "subscription_tier"
            ],
            "text_fields": ["title", "abstract", "key_findings", "methodology"],
            "summary_prompt": """Summarize this wealth management report for an investor in 2-3 sentences.
Focus on: key investment thesis, target investor profile, risk considerations, and actionable insights.
Report: {item_text}""",
        },
        Domain.ECOMMERCE: {
            "item_name": "product",
            "item_id_field": "product_id",
            "features": [
                "price", "discount_percent", "rating", "review_count",
                "stock_status", "shipping_days", "return_policy_score", "brand_popularity"
            ],
            "context_features": [
                "search_query", "category", "price_range", "brand_filter",
                "sort_preference", "device_type", "session_duration"
            ],
            "user_features": [
                "user_id", "purchase_history_count", "cart_abandonment_rate",
                "avg_order_value", "preferred_categories", "days_since_last_purchase"
            ],
            "text_fields": ["name", "description", "specifications", "reviews_summary"],
            "summary_prompt": """Summarize this product for a shopper in 2-3 sentences.
Focus on: key features, best use case, value proposition, and standout qualities.
Product: {item_text}""",
        }
    }
    
    @classmethod
    def get(cls, domain: Domain) -> dict:
        """Get configuration for a specific domain."""
        return cls.CONFIGS[domain]
    
    @classmethod
    def get_features(cls, domain: Domain) -> list[str]:
        """Get all feature names for a domain."""
        config = cls.CONFIGS[domain]
        return config["features"] + config["context_features"] + config["user_features"]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
