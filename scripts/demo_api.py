#!/usr/bin/env python3
"""
Demo script to test the CusFlow API.

Usage:
    python scripts/demo_api.py --host localhost --port 8000
"""

import argparse
import json

import httpx


def main():
    parser = argparse.ArgumentParser(description="Demo CusFlow API")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print("=" * 60)
    print("CusFlow API Demo")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    
    with httpx.Client(timeout=30.0) as client:
        # Health check
        print("\n1️⃣ Health Check")
        print("-" * 40)
        try:
            resp = client.get(f"{base_url}/health")
            print(f"Status: {resp.status_code}")
            print(json.dumps(resp.json(), indent=2))
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure the API is running: python -m src.cli serve")
            return
        
        # Create some test items
        print("\n2️⃣ Creating Test Items")
        print("-" * 40)
        
        test_items = [
            {
                "item_id": "hotel_001",
                "domain": "hotel",
                "name": "Grand Plaza Hotel",
                "description": "Luxury 5-star hotel in downtown",
                "features": {
                    "features": {"star_rating": 5, "review_score": 4.8, "price_per_night": 250},
                    "popularity_score": 0.9,
                    "quality_score": 0.95,
                }
            },
            {
                "item_id": "hotel_002",
                "domain": "hotel",
                "name": "Budget Inn Express",
                "description": "Affordable accommodation near airport",
                "features": {
                    "features": {"star_rating": 3, "review_score": 4.0, "price_per_night": 80},
                    "popularity_score": 0.7,
                    "quality_score": 0.75,
                }
            },
            {
                "item_id": "hotel_003",
                "domain": "hotel",
                "name": "Seaside Resort & Spa",
                "description": "Beachfront resort with spa facilities",
                "features": {
                    "features": {"star_rating": 4, "review_score": 4.5, "price_per_night": 180},
                    "popularity_score": 0.85,
                    "quality_score": 0.88,
                }
            },
        ]
        
        try:
            resp = client.post(f"{base_url}/items/batch", json=test_items)
            print(f"Status: {resp.status_code}")
            print(json.dumps(resp.json(), indent=2))
        except Exception as e:
            print(f"Error: {e}")
        
        # Create a test user
        print("\n3️⃣ Creating Test User")
        print("-" * 40)
        
        test_user = {
            "user_id": "user_001",
            "features": {
                "features": {"booking_history_count": 5, "avg_spend": 150, "preferred_star_rating": 4},
                "total_interactions": 20,
            },
            "segments": ["frequent_traveler"],
        }
        
        try:
            resp = client.post(f"{base_url}/users", json=test_user)
            print(f"Status: {resp.status_code}")
            print(json.dumps(resp.json(), indent=2))
        except Exception as e:
            print(f"Error: {e}")
        
        # Get ranking
        print("\n4️⃣ Getting Personalized Rankings")
        print("-" * 40)
        
        ranking_request = {
            "user_id": "user_001",
            "candidate_ids": ["hotel_001", "hotel_002", "hotel_003"],
            "num_results": 3,
            "context": {
                "device_type": "desktop",
                "timestamp": "2024-01-15T10:30:00Z",
            },
        }
        
        try:
            resp = client.post(f"{base_url}/rank", json=ranking_request)
            print(f"Status: {resp.status_code}")
            result = resp.json()
            
            print(f"\nRequest ID: {result.get('request_id')}")
            print(f"Latency: {result.get('latency_ms', 0):.2f}ms")
            print(f"Model: {result.get('model_version')}")
            print("\nRanked Items:")
            
            for item in result.get("items", []):
                print(f"  {item['position']}. {item['item_id']} (score: {item['score']:.4f})")
                if item.get("item"):
                    print(f"     - {item['item']['name']}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Get item details
        print("\n5️⃣ Getting Item Details")
        print("-" * 40)
        
        try:
            resp = client.get(f"{base_url}/items/hotel_001")
            print(f"Status: {resp.status_code}")
            print(json.dumps(resp.json(), indent=2))
        except Exception as e:
            print(f"Error: {e}")
        
        # Get config
        print("\n6️⃣ Getting Configuration")
        print("-" * 40)
        
        try:
            resp = client.get(f"{base_url}/admin/config")
            print(f"Status: {resp.status_code}")
            print(json.dumps(resp.json(), indent=2))
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
