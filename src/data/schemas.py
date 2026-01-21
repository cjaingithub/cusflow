"""
Pydantic schemas for data models.

These schemas define the structure of items, users, features, and requests
across all supported domains (hotels, wealth reports, e-commerce).
"""

from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ItemFeatures(BaseModel):
    """Features associated with an item (hotel, report, product)."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Core features (domain-specific values)
    features: dict[str, float | int | str] = Field(default_factory=dict)
    
    # GenAI-generated features
    summary: str | None = Field(default=None, description="LLM-generated summary")
    embedding: list[float] | None = Field(default=None, description="Dense embedding vector")
    
    # Computed features
    popularity_score: float = Field(default=0.0, description="Historical popularity")
    freshness_score: float = Field(default=0.0, description="Recency-based score")
    quality_score: float = Field(default=0.0, description="Quality/rating composite")
    
    @field_validator("embedding", mode="before")
    @classmethod
    def convert_embedding(cls, v: Any) -> list[float] | None:
        if v is None:
            return None
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v


class Item(BaseModel):
    """Represents an item in the catalog (hotel, report, product)."""
    
    item_id: str = Field(..., description="Unique item identifier")
    domain: str = Field(..., description="Domain type: hotel, wealth_report, ecommerce")
    
    # Text content for GenAI processing
    name: str = Field(..., description="Item name/title")
    description: str = Field(default="", description="Full description")
    text_content: dict[str, str] = Field(
        default_factory=dict, 
        description="Additional text fields for embedding"
    )
    
    # Features
    features: ItemFeatures = Field(default_factory=ItemFeatures)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    
    def get_text_for_embedding(self) -> str:
        """Combine all text fields for embedding generation."""
        texts = [self.name, self.description]
        texts.extend(self.text_content.values())
        return " ".join(filter(None, texts))


class UserFeatures(BaseModel):
    """Features associated with a user."""
    
    # Behavioral features
    features: dict[str, float | int | str] = Field(default_factory=dict)
    
    # Preference embedding (learned from history)
    preference_embedding: list[float] | None = Field(
        default=None, 
        description="User preference embedding"
    )
    
    # Aggregated statistics
    total_interactions: int = Field(default=0)
    avg_rating_given: float = Field(default=0.0)
    conversion_rate: float = Field(default=0.0)
    
    @field_validator("preference_embedding", mode="before")
    @classmethod
    def convert_embedding(cls, v: Any) -> list[float] | None:
        if v is None:
            return None
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v


class User(BaseModel):
    """Represents a user in the system."""
    
    user_id: str = Field(..., description="Unique user identifier")
    
    # Features
    features: UserFeatures = Field(default_factory=UserFeatures)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Segments
    segments: list[str] = Field(default_factory=list, description="User segments")


class ContextFeatures(BaseModel):
    """Contextual features for a ranking request."""
    
    # Request context
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    device_type: str = Field(default="desktop", description="web, mobile, tablet")
    platform: str = Field(default="web", description="web, ios, android")
    
    # Session context
    session_id: str | None = Field(default=None)
    page_number: int = Field(default=1)
    items_per_page: int = Field(default=20)
    
    # Domain-specific context (stored as dict for flexibility)
    context: dict[str, Any] = Field(default_factory=dict)
    
    # Search/filter context
    query: str | None = Field(default=None, description="Search query if applicable")
    filters: dict[str, Any] = Field(default_factory=dict, description="Applied filters")
    sort_preference: str | None = Field(default=None)


class FeedbackEvent(BaseModel):
    """User feedback event for training and evaluation."""
    
    event_id: str = Field(..., description="Unique event identifier")
    user_id: str = Field(..., description="User who generated the event")
    item_id: str = Field(..., description="Item interacted with")
    
    # Event type and value
    event_type: str = Field(
        ..., 
        description="click, impression, add_to_cart, purchase, rating, etc."
    )
    event_value: float = Field(
        default=1.0, 
        description="Event value (e.g., rating, purchase amount)"
    )
    
    # Position bias information
    position: int | None = Field(default=None, description="Position in the list when shown")
    
    # Context at event time
    context: ContextFeatures = Field(default_factory=ContextFeatures)
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # For training labels
    relevance_label: int | None = Field(
        default=None, 
        description="Relevance grade (0-4) for LTR training"
    )


class RankingRequest(BaseModel):
    """Request to the ranking API."""
    
    # User information
    user_id: str = Field(..., description="User requesting recommendations")
    user_features: UserFeatures | None = Field(
        default=None, 
        description="Optional user features override"
    )
    
    # Candidate items (optional - will retrieve if not provided)
    candidate_ids: list[str] | None = Field(
        default=None, 
        description="Pre-selected candidate item IDs"
    )
    
    # Context
    context: ContextFeatures = Field(default_factory=ContextFeatures)
    
    # Ranking parameters
    num_results: int = Field(default=20, ge=1, le=100)
    
    # Experiment configuration
    experiment_id: str | None = Field(default=None, description="A/B test experiment ID")
    treatment_group: str | None = Field(default=None, description="control or treatment")
    
    # Feature flags
    use_genai_features: bool = Field(default=True, description="Include GenAI features")
    apply_diversity: bool = Field(default=False, description="Apply result diversification")


class RankedItem(BaseModel):
    """An item with its ranking score."""
    
    item_id: str
    score: float = Field(..., description="Ranking score")
    position: int = Field(..., description="Position in ranked list (1-indexed)")
    
    # Feature contributions (for explainability)
    feature_contributions: dict[str, float] = Field(
        default_factory=dict,
        description="Feature importance for this prediction"
    )
    
    # Optional item details
    item: Item | None = Field(default=None)


class RankingResponse(BaseModel):
    """Response from the ranking API."""
    
    # Request metadata
    request_id: str = Field(..., description="Unique request identifier")
    user_id: str
    
    # Ranked results
    items: list[RankedItem] = Field(default_factory=list)
    
    # Timing
    latency_ms: float = Field(..., description="Total latency in milliseconds")
    
    # Model information
    model_version: str = Field(default="v1", description="Ranking model version")
    
    # Experiment info
    experiment_id: str | None = None
    treatment_group: str | None = None


class TrainingExample(BaseModel):
    """A single training example for LTR model."""
    
    query_id: str = Field(..., description="Query/request identifier for grouping")
    item_id: str
    
    # Features vector
    features: list[float] = Field(..., description="Feature vector")
    feature_names: list[str] = Field(default_factory=list)
    
    # Label
    relevance: int = Field(..., ge=0, le=4, description="Relevance grade")
    
    # Position for bias correction
    position: int | None = Field(default=None)
    
    # Propensity for inverse propensity weighting
    propensity: float = Field(default=1.0, description="Display propensity for IPW")


class EvaluationResult(BaseModel):
    """Results from offline evaluation."""
    
    # Evaluation metadata
    eval_id: str
    model_version: str
    dataset_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Metrics
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Metric name -> value mapping"
    )
    
    # Breakdown by segment
    segment_metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Segment -> metrics mapping"
    )
    
    # Statistical significance
    confidence_intervals: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description="95% CI for each metric"
    )
    
    # Sample size
    num_queries: int
    num_items: int


class ABTestResult(BaseModel):
    """Results from A/B simulation."""
    
    experiment_id: str
    control_model: str
    treatment_model: str
    
    # Traffic
    control_traffic: int
    treatment_traffic: int
    
    # Primary metrics
    control_metrics: dict[str, float]
    treatment_metrics: dict[str, float]
    
    # Uplift
    relative_uplift: dict[str, float] = Field(
        default_factory=dict,
        description="Relative improvement for each metric"
    )
    
    # Statistical significance
    p_values: dict[str, float] = Field(default_factory=dict)
    is_significant: dict[str, bool] = Field(default_factory=dict)
    
    # Simulation metadata
    simulation_date: datetime = Field(default_factory=datetime.utcnow)
    replay_start_date: datetime | None = None
    replay_end_date: datetime | None = None
