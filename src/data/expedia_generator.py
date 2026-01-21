"""
Expedia-like Hotel Data Generator

Generates realistic hotel search data with:
- Rich property attributes (40+ features)
- Dynamic pricing signals
- User behavioral patterns
- Position bias simulation
"""

import random
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np

from src.data.schemas import FeedbackEvent, Item, ItemFeatures, TrainingExample, User, UserFeatures


class ExpediaDataGenerator:
    """Generate Expedia-like hotel data."""
    
    DESTINATIONS = [("New York", "NYC"), ("Los Angeles", "LAX"), ("San Francisco", "SFO"), 
                   ("Miami", "MIA"), ("Las Vegas", "LAS"), ("Chicago", "ORD")]
    CHAINS = [("Marriott", 0.85, 180), ("Hilton", 0.83, 170), ("Hyatt", 0.82, 175),
              ("Budget Inn", 0.65, 80), ("Luxury Collection", 0.95, 350)]
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
    
    def generate_hotels(self, n_hotels: int = 500) -> list[Item]:
        hotels = []
        for i in range(n_hotels):
            dest, code = random.choice(self.DESTINATIONS)
            chain, quality, price_base = random.choice(self.CHAINS)
            star = 5 if quality >= 0.9 else (4 if quality >= 0.75 else 3)
            
            review_score = min(10, max(5, quality * 8 + self.rng.normal(0, 0.5)))
            price = max(50, price_base * (1 + self.rng.normal(0, 0.2)))
            
            features = {
                "star_rating": star, "review_score": round(review_score, 1),
                "review_count": int(self.rng.poisson(100)), "price_per_night": round(price, 2),
                "location_score": round(self.rng.uniform(6, 10), 1),
                "distance_to_center_km": round(self.rng.exponential(3), 2),
                "destination": dest, "chain": chain,
                "has_pool": random.random() < 0.6, "has_spa": random.random() < 0.4,
                "has_free_wifi": True, "has_free_parking": random.random() < 0.5,
                "free_cancellation": random.random() < 0.7,
                "amenity_count": int(star * 3 + self.rng.poisson(2)),
                "bookings_last_24h": int(self.rng.poisson(5)),
                "rooms_available": int(self.rng.integers(1, 20)),
                "value_score": round(review_score / (price / 100), 2),
            }
            
            hotels.append(Item(
                item_id=f"hotel_{i:06d}", domain="hotel",
                name=f"{chain} {dest}", description=f"{star}-star hotel in {dest}",
                features=ItemFeatures(features=features, popularity_score=quality, quality_score=review_score/10)))
        return hotels
    
    def generate_users(self, n_users: int = 1000) -> list[User]:
        users = []
        for i in range(n_users):
            segment = self.rng.choice(["budget", "mid_range", "premium", "luxury"], p=[0.3, 0.4, 0.2, 0.1])
            avg_spend = {"luxury": 350, "premium": 200, "mid_range": 120, "budget": 70}[segment]
            star_pref = {"luxury": 5, "premium": 4, "mid_range": 4, "budget": 3}[segment]
            
            features = {
                "segment": segment, "avg_spend_per_night": avg_spend + self.rng.normal(0, 30),
                "preferred_star_rating": star_pref,
                "price_sensitivity": {"luxury": 0.2, "premium": 0.4, "mid_range": 0.6, "budget": 0.85}[segment],
                "booking_history_count": int(self.rng.poisson(5)),
            }
            users.append(User(user_id=f"user_{i:06d}", features=UserFeatures(features=features), segments=[segment]))
        return users
    
    def generate_search_sessions(self, users: list[User], hotels: list[Item], n_sessions: int = 5000) -> list[FeedbackEvent]:
        events = []
        hotels_by_dest = {}
        for h in hotels:
            d = h.features.features["destination"]
            hotels_by_dest.setdefault(d, []).append(h)
        
        for sid in range(n_sessions):
            user = random.choice(users)
            dest = random.choice(list(hotels_by_dest.keys()))
            candidates = random.sample(hotels_by_dest[dest], min(20, len(hotels_by_dest[dest])))
            
            for pos, hotel in enumerate(candidates, 1):
                exam_prob = 1 / (1 + 0.2 * pos)
                if random.random() > exam_prob:
                    continue
                
                h = hotel.features.features
                u = user.features.features
                click_prob = 0.15
                if h["star_rating"] == u["preferred_star_rating"]:
                    click_prob += 0.1
                if abs(h["price_per_night"] - u["avg_spend_per_night"]) < 50:
                    click_prob += 0.05
                
                if random.random() < click_prob:
                    events.append(FeedbackEvent(
                        event_id=f"evt_{len(events)}", user_id=user.user_id, item_id=hotel.item_id,
                        event_type="click", event_value=1, position=pos, timestamp=datetime.now(),
                        relevance_label=2 + (1 if random.random() < 0.1 else 0)))
        return events
    
    def generate_training_data(self, hotels: list[Item], users: list[User], events: list[FeedbackEvent]) -> Iterator[TrainingExample]:
        hotel_lookup = {h.item_id: h for h in hotels}
        user_lookup = {u.user_id: u for u in users}
        feature_names = ["star_rating", "review_score", "price_per_night", "location_score", 
                        "amenity_count", "value_score", "price_diff", "star_match"]
        
        from collections import Counter
        sessions = {}
        for e in events:
            sessions.setdefault(e.user_id, []).append(e)
        
        qid = 0
        for uid, evts in sessions.items():
            qid += 1
            user = user_lookup.get(uid)
            if not user:
                continue
            u = user.features.features
            
            for e in evts:
                h = hotel_lookup.get(e.item_id)
                if not h:
                    continue
                hf = h.features.features
                features = [
                    hf["star_rating"], hf["review_score"], hf["price_per_night"],
                    hf["location_score"], hf["amenity_count"], hf["value_score"],
                    hf["price_per_night"] - u["avg_spend_per_night"],
                    float(hf["star_rating"] == u["preferred_star_rating"]),
                ]
                yield TrainingExample(query_id=f"q_{qid}", item_id=e.item_id, features=features,
                                      feature_names=feature_names, relevance=e.relevance_label or 0,
                                      position=e.position, propensity=1/(1+0.2*e.position))
