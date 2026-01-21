"""
Data loading and synthetic data generation.

Supports loading from CSV, Parquet, or generating synthetic data
for testing and demonstration across different domains.
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import polars as pl

from src.config import Domain, DomainConfig, get_settings
from src.data.schemas import (
    ContextFeatures,
    FeedbackEvent,
    Item,
    ItemFeatures,
    TrainingExample,
    User,
    UserFeatures,
)


class DataLoader:
    """Load data from various file formats."""
    
    def __init__(self, data_path: Path | None = None):
        self.settings = get_settings()
        self.data_path = data_path or self.settings.data_path
    
    def load_items(self, filename: str = "items.parquet") -> list[Item]:
        """Load items from file."""
        filepath = self.data_path / filename
        
        if filepath.suffix == ".parquet":
            df = pl.read_parquet(filepath)
        elif filepath.suffix == ".csv":
            df = pl.read_csv(filepath)
        elif filepath.suffix == ".json":
            with open(filepath) as f:
                data = json.load(f)
            return [Item(**item) for item in data]
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return self._df_to_items(df)
    
    def load_users(self, filename: str = "users.parquet") -> list[User]:
        """Load users from file."""
        filepath = self.data_path / filename
        
        if filepath.suffix == ".parquet":
            df = pl.read_parquet(filepath)
        elif filepath.suffix == ".csv":
            df = pl.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return self._df_to_users(df)
    
    def load_events(self, filename: str = "events.parquet") -> list[FeedbackEvent]:
        """Load feedback events from file."""
        filepath = self.data_path / filename
        
        if filepath.suffix == ".parquet":
            df = pl.read_parquet(filepath)
        elif filepath.suffix == ".csv":
            df = pl.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return self._df_to_events(df)
    
    def load_training_data(
        self, 
        filename: str = "training.parquet"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training data for LTR model.
        
        Returns:
            features: (n_samples, n_features) array
            labels: (n_samples,) array of relevance grades
            groups: (n_queries,) array of group sizes
        """
        filepath = self.data_path / filename
        
        if filepath.suffix == ".parquet":
            df = pl.read_parquet(filepath)
        else:
            df = pl.read_csv(filepath)
        
        # Extract features (columns starting with 'f_')
        feature_cols = [c for c in df.columns if c.startswith("f_")]
        features = df.select(feature_cols).to_numpy()
        
        # Labels
        labels = df["relevance"].to_numpy()
        
        # Groups (for LambdaMART)
        groups = df.group_by("query_id", maintain_order=True).agg(
            pl.count()
        )["count"].to_numpy()
        
        return features, labels, groups
    
    def _df_to_items(self, df: pl.DataFrame) -> list[Item]:
        """Convert DataFrame to list of Item objects."""
        items = []
        for row in df.iter_rows(named=True):
            item_features = ItemFeatures(
                features={k: v for k, v in row.items() 
                         if k not in ["item_id", "domain", "name", "description"]},
            )
            items.append(Item(
                item_id=str(row["item_id"]),
                domain=row.get("domain", self.settings.domain.value),
                name=row.get("name", ""),
                description=row.get("description", ""),
                features=item_features,
            ))
        return items
    
    def _df_to_users(self, df: pl.DataFrame) -> list[User]:
        """Convert DataFrame to list of User objects."""
        users = []
        for row in df.iter_rows(named=True):
            user_features = UserFeatures(
                features={k: v for k, v in row.items() if k != "user_id"},
            )
            users.append(User(
                user_id=str(row["user_id"]),
                features=user_features,
            ))
        return users
    
    def _df_to_events(self, df: pl.DataFrame) -> list[FeedbackEvent]:
        """Convert DataFrame to list of FeedbackEvent objects."""
        events = []
        for row in df.iter_rows(named=True):
            events.append(FeedbackEvent(
                event_id=str(row.get("event_id", uuid.uuid4())),
                user_id=str(row["user_id"]),
                item_id=str(row["item_id"]),
                event_type=row.get("event_type", "click"),
                event_value=float(row.get("event_value", 1.0)),
                position=row.get("position"),
                timestamp=row.get("timestamp", datetime.utcnow()),
            ))
        return events


class SyntheticDataGenerator:
    """
    Generate synthetic data for testing and demonstration.
    
    Supports multiple domains with realistic feature distributions.
    """
    
    # Hotel name components
    HOTEL_PREFIXES = ["Grand", "Royal", "Paradise", "Sunset", "Ocean", "Mountain", 
                     "City", "Metro", "Luxury", "Budget", "Boutique", "Historic"]
    HOTEL_TYPES = ["Hotel", "Resort", "Inn", "Suites", "Lodge", "Palace", "Plaza"]
    HOTEL_LOCATIONS = ["Downtown", "Beachfront", "Airport", "Convention Center", 
                       "Old Town", "Business District", "Waterfront"]
    
    # Wealth report components
    REPORT_TOPICS = ["Equity", "Fixed Income", "Alternative Investments", "Real Estate",
                    "Commodities", "Cryptocurrency", "ESG", "Emerging Markets"]
    REPORT_TYPES = ["Deep Dive", "Quarterly Review", "Strategy Note", "Market Outlook",
                   "Sector Analysis", "Risk Assessment", "Portfolio Construction"]
    
    # E-commerce components
    PRODUCT_CATEGORIES = ["Electronics", "Fashion", "Home & Garden", "Sports", 
                         "Beauty", "Books", "Toys", "Automotive"]
    
    def __init__(self, domain: Domain | None = None, seed: int = 42):
        self.settings = get_settings()
        self.domain = domain or self.settings.domain
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        self.domain_config = DomainConfig.get(self.domain)
    
    def generate_items(self, n_items: int = 1000) -> list[Item]:
        """Generate synthetic items for the configured domain."""
        if self.domain == Domain.HOTEL:
            return self._generate_hotels(n_items)
        elif self.domain == Domain.WEALTH_REPORT:
            return self._generate_reports(n_items)
        elif self.domain == Domain.ECOMMERCE:
            return self._generate_products(n_items)
        else:
            raise ValueError(f"Unsupported domain: {self.domain}")
    
    def _generate_hotels(self, n_items: int) -> list[Item]:
        """Generate synthetic hotel data."""
        items = []
        
        for i in range(n_items):
            # Generate hotel name
            prefix = random.choice(self.HOTEL_PREFIXES)
            hotel_type = random.choice(self.HOTEL_TYPES)
            location = random.choice(self.HOTEL_LOCATIONS)
            name = f"The {prefix} {hotel_type}"
            
            # Generate features with realistic correlations
            star_rating = self.rng.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
            base_price = 50 + star_rating * 40 + self.rng.normal(0, 20)
            
            # Quality correlates with stars and price
            review_score = min(5.0, max(1.0, 
                star_rating * 0.6 + self.rng.normal(0.5, 0.3) + base_price / 200
            ))
            
            features = {
                "star_rating": star_rating,
                "review_score": round(review_score, 1),
                "price_per_night": max(30, round(base_price, 2)),
                "location_score": round(self.rng.uniform(3, 5), 1),
                "amenities_count": int(5 + star_rating * 3 + self.rng.poisson(2)),
                "distance_to_center": round(self.rng.exponential(3), 1),
                "booking_count": int(self.rng.poisson(100 * (star_rating / 3))),
                "cancellation_policy": random.choice([0, 1, 2]),  # strict, moderate, flexible
            }
            
            description = (
                f"A beautiful {star_rating}-star {hotel_type.lower()} located in {location}. "
                f"Features {features['amenities_count']} amenities including WiFi, pool, and gym. "
                f"Just {features['distance_to_center']}km from the city center with excellent "
                f"transport links. Rated {features['review_score']}/5 by our guests."
            )
            
            items.append(Item(
                item_id=f"hotel_{i:05d}",
                domain="hotel",
                name=name,
                description=description,
                text_content={
                    "location_description": f"Located in {location}, perfect for business and leisure.",
                    "amenities_text": "Free WiFi, Swimming Pool, Fitness Center, Restaurant, Spa",
                },
                features=ItemFeatures(
                    features=features,
                    popularity_score=features["booking_count"] / 500,
                    quality_score=features["review_score"] / 5,
                ),
            ))
        
        return items
    
    def _generate_reports(self, n_items: int) -> list[Item]:
        """Generate synthetic wealth management reports."""
        items = []
        
        for i in range(n_items):
            topic = random.choice(self.REPORT_TOPICS)
            report_type = random.choice(self.REPORT_TYPES)
            
            # Risk level affects other features
            risk_level = self.rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1])
            
            features = {
                "asset_class": topic,
                "risk_level": risk_level,
                "return_potential": round(2 + risk_level * 1.5 + self.rng.normal(0, 1), 1),
                "time_horizon": random.choice([1, 3, 5, 10]),  # years
                "min_investment": random.choice([1000, 10000, 50000, 100000, 500000]),
                "publication_date_recency": int(self.rng.exponential(30)),  # days ago
                "author_reputation_score": round(self.rng.uniform(3, 5), 1),
                "download_count": int(self.rng.poisson(500)),
                "citation_count": int(self.rng.poisson(20)),
            }
            
            name = f"{topic} {report_type}: Q{random.randint(1, 4)} 2024"
            description = (
                f"Comprehensive {report_type.lower()} on {topic.lower()} markets. "
                f"Risk Level: {risk_level}/5. Expected return potential: {features['return_potential']}%. "
                f"Suitable for investors with a {features['time_horizon']}-year time horizon."
            )
            
            items.append(Item(
                item_id=f"report_{i:05d}",
                domain="wealth_report",
                name=name,
                description=description,
                text_content={
                    "abstract": f"This report analyzes {topic.lower()} opportunities...",
                    "key_findings": f"Key finding: {topic} sector shows promising growth...",
                },
                features=ItemFeatures(
                    features=features,
                    popularity_score=features["download_count"] / 1000,
                    freshness_score=max(0, 1 - features["publication_date_recency"] / 90),
                    quality_score=features["author_reputation_score"] / 5,
                ),
            ))
        
        return items
    
    def _generate_products(self, n_items: int) -> list[Item]:
        """Generate synthetic e-commerce products."""
        items = []
        
        for i in range(n_items):
            category = random.choice(self.PRODUCT_CATEGORIES)
            
            # Base price varies by category
            base_prices = {
                "Electronics": 200, "Fashion": 50, "Home & Garden": 100,
                "Sports": 80, "Beauty": 30, "Books": 20, "Toys": 40, "Automotive": 150
            }
            base_price = base_prices.get(category, 50)
            price = max(5, base_price + self.rng.normal(0, base_price * 0.5))
            
            features = {
                "price": round(price, 2),
                "discount_percent": random.choice([0, 0, 0, 10, 15, 20, 25, 30]),
                "rating": round(max(1, min(5, self.rng.normal(4, 0.5))), 1),
                "review_count": int(self.rng.poisson(100)),
                "stock_status": random.choice([0, 1, 1, 1, 2]),  # out, low, in stock
                "shipping_days": random.choice([1, 2, 3, 5, 7]),
                "return_policy_score": random.choice([1, 2, 3]),  # strict to easy
                "brand_popularity": round(self.rng.uniform(1, 5), 1),
            }
            
            name = f"Premium {category} Item #{i}"
            description = (
                f"High-quality {category.lower()} product. "
                f"Rated {features['rating']}/5 stars based on {features['review_count']} reviews. "
                f"Ships in {features['shipping_days']} days."
            )
            
            items.append(Item(
                item_id=f"product_{i:05d}",
                domain="ecommerce",
                name=name,
                description=description,
                features=ItemFeatures(
                    features=features,
                    popularity_score=features["review_count"] / 200,
                    quality_score=features["rating"] / 5,
                ),
            ))
        
        return items
    
    def generate_users(self, n_users: int = 500) -> list[User]:
        """Generate synthetic users."""
        users = []
        
        for i in range(n_users):
            if self.domain == Domain.HOTEL:
                features = {
                    "booking_history_count": int(self.rng.poisson(5)),
                    "avg_spend": round(100 + self.rng.exponential(100), 2),
                    "preferred_star_rating": self.rng.choice([3, 4, 5], p=[0.3, 0.5, 0.2]),
                    "loyalty_tier": random.choice([0, 1, 2, 3]),
                    "days_since_last_booking": int(self.rng.exponential(60)),
                }
            elif self.domain == Domain.WEALTH_REPORT:
                features = {
                    "aum": random.choice([50000, 100000, 500000, 1000000, 5000000]),
                    "investment_experience_years": int(self.rng.exponential(5)),
                    "preferred_asset_classes": random.sample(self.REPORT_TOPICS, k=2),
                    "read_history_count": int(self.rng.poisson(20)),
                    "subscription_tier": random.choice([0, 1, 2]),
                }
            else:  # ecommerce
                features = {
                    "purchase_history_count": int(self.rng.poisson(10)),
                    "cart_abandonment_rate": round(self.rng.beta(2, 5), 2),
                    "avg_order_value": round(50 + self.rng.exponential(50), 2),
                    "preferred_categories": random.sample(self.PRODUCT_CATEGORIES, k=3),
                    "days_since_last_purchase": int(self.rng.exponential(30)),
                }
            
            users.append(User(
                user_id=f"user_{i:05d}",
                features=UserFeatures(
                    features=features,
                    total_interactions=int(self.rng.poisson(50)),
                ),
                segments=self._assign_segments(),
            ))
        
        return users
    
    def _assign_segments(self) -> list[str]:
        """Assign user segments based on random selection."""
        segments = []
        if random.random() < 0.3:
            segments.append("high_value")
        if random.random() < 0.2:
            segments.append("new_user")
        if random.random() < 0.4:
            segments.append("frequent_browser")
        if random.random() < 0.15:
            segments.append("deal_seeker")
        return segments
    
    def generate_events(
        self, 
        users: list[User], 
        items: list[Item],
        n_events: int = 10000,
        start_date: datetime | None = None,
    ) -> list[FeedbackEvent]:
        """
        Generate synthetic feedback events with realistic patterns.
        
        Includes position bias simulation for click data.
        """
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=90)
        
        events = []
        item_ids = [item.item_id for item in items]
        user_ids = [user.user_id for user in users]
        
        # Create item quality scores for click probability
        item_quality = {
            item.item_id: item.features.quality_score + item.features.popularity_score
            for item in items
        }
        
        for i in range(n_events):
            user_id = random.choice(user_ids)
            
            # Simulate a ranking session
            session_items = random.sample(item_ids, k=min(20, len(item_ids)))
            
            for position, item_id in enumerate(session_items, 1):
                # Position bias: lower positions get examined less
                examination_prob = 1 / (position ** 0.5)
                
                if random.random() < examination_prob:
                    # Item was examined, now check if clicked
                    click_prob = item_quality.get(item_id, 0.5) * 0.5
                    
                    if random.random() < click_prob:
                        # Determine event type based on funnel
                        event_roll = random.random()
                        if event_roll < 0.1:  # 10% convert
                            event_type = "purchase"
                            event_value = items[int(item_id.split("_")[1])].features.features.get(
                                "price_per_night", 
                                items[int(item_id.split("_")[1])].features.features.get("price", 100)
                            )
                            relevance = 4
                        elif event_roll < 0.3:  # 20% add to cart
                            event_type = "add_to_cart"
                            event_value = 1.0
                            relevance = 3
                        else:  # 70% just click
                            event_type = "click"
                            event_value = 1.0
                            relevance = 2
                        
                        events.append(FeedbackEvent(
                            event_id=f"event_{len(events):08d}",
                            user_id=user_id,
                            item_id=item_id,
                            event_type=event_type,
                            event_value=event_value,
                            position=position,
                            timestamp=start_date + timedelta(
                                seconds=random.randint(0, 90 * 24 * 3600)
                            ),
                            relevance_label=relevance,
                        ))
                    else:
                        # Impression without click
                        if random.random() < 0.3:  # Log 30% of impressions
                            events.append(FeedbackEvent(
                                event_id=f"event_{len(events):08d}",
                                user_id=user_id,
                                item_id=item_id,
                                event_type="impression",
                                event_value=0.0,
                                position=position,
                                timestamp=start_date + timedelta(
                                    seconds=random.randint(0, 90 * 24 * 3600)
                                ),
                                relevance_label=1,
                            ))
        
        return events
    
    def generate_training_data(
        self,
        users: list[User],
        items: list[Item],
        events: list[FeedbackEvent],
        feature_names: list[str] | None = None,
    ) -> Iterator[TrainingExample]:
        """
        Generate training examples from events for LTR model.
        
        Groups by query (user session) and creates feature vectors.
        """
        if feature_names is None:
            feature_names = self.domain_config["features"]
        
        # Create item feature lookup
        item_features = {item.item_id: item.features.features for item in items}
        
        # Group events by user for query groups
        from collections import defaultdict
        user_events = defaultdict(list)
        for event in events:
            user_events[event.user_id].append(event)
        
        query_id = 0
        for user_id, user_event_list in user_events.items():
            query_id += 1
            
            for event in user_event_list:
                item_feats = item_features.get(event.item_id, {})
                
                # Build feature vector
                features = []
                for fname in feature_names:
                    val = item_feats.get(fname, 0)
                    # Convert non-numeric to 0
                    if isinstance(val, str):
                        val = hash(val) % 100 / 100  # Simple encoding
                    elif isinstance(val, list):
                        val = len(val)
                    features.append(float(val))
                
                yield TrainingExample(
                    query_id=f"q_{query_id:06d}",
                    item_id=event.item_id,
                    features=features,
                    feature_names=feature_names,
                    relevance=event.relevance_label or 0,
                    position=event.position,
                    propensity=1 / (event.position ** 0.5) if event.position else 1.0,
                )
    
    def save_synthetic_data(
        self,
        output_path: Path | None = None,
        n_items: int = 1000,
        n_users: int = 500,
        n_events: int = 10000,
    ) -> dict[str, Path]:
        """Generate and save synthetic dataset."""
        output_path = output_path or self.settings.data_path
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate data
        items = self.generate_items(n_items)
        users = self.generate_users(n_users)
        events = self.generate_events(users, items, n_events)
        
        # Convert to DataFrames and save
        items_df = pl.DataFrame([
            {
                "item_id": item.item_id,
                "domain": item.domain,
                "name": item.name,
                "description": item.description,
                **item.features.features,
            }
            for item in items
        ])
        
        users_df = pl.DataFrame([
            {
                "user_id": user.user_id,
                **{k: str(v) if isinstance(v, list) else v 
                   for k, v in user.features.features.items()},
            }
            for user in users
        ])
        
        events_df = pl.DataFrame([
            {
                "event_id": event.event_id,
                "user_id": event.user_id,
                "item_id": event.item_id,
                "event_type": event.event_type,
                "event_value": event.event_value,
                "position": event.position,
                "timestamp": event.timestamp,
                "relevance_label": event.relevance_label,
            }
            for event in events
        ])
        
        # Save files
        items_path = output_path / "items.parquet"
        users_path = output_path / "users.parquet"
        events_path = output_path / "events.parquet"
        
        items_df.write_parquet(items_path)
        users_df.write_parquet(users_path)
        events_df.write_parquet(events_path)
        
        # Also save training data
        training_examples = list(self.generate_training_data(users, items, events))
        training_df = pl.DataFrame([
            {
                "query_id": ex.query_id,
                "item_id": ex.item_id,
                **{f"f_{name}": val for name, val in zip(ex.feature_names, ex.features)},
                "relevance": ex.relevance,
                "position": ex.position,
                "propensity": ex.propensity,
            }
            for ex in training_examples
        ])
        training_path = output_path / "training.parquet"
        training_df.write_parquet(training_path)
        
        return {
            "items": items_path,
            "users": users_path,
            "events": events_path,
            "training": training_path,
        }
