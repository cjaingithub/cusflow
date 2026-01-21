#!/usr/bin/env python3
"""
Load features into Redis feature store.

Usage:
    python scripts/load_features.py --data data/ --redis-host localhost
"""

import argparse
from pathlib import Path

from src.data.loaders import DataLoader, SyntheticDataGenerator
from src.config import Domain
from src.store.redis_store import RedisFeatureStore


def main():
    parser = argparse.ArgumentParser(description="Load features into Redis")
    parser.add_argument("--data", type=Path, default=Path("data/"))
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data first")
    parser.add_argument("--embeddings", action="store_true", help="Generate and store embeddings")
    parser.add_argument("--domain", type=str, default="hotel")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CusFlow Feature Loading")
    print("=" * 60)
    
    # Generate data if requested
    if args.generate:
        print(f"\n📊 Generating synthetic data for domain: {args.domain}")
        generator = SyntheticDataGenerator(domain=Domain(args.domain))
        generator.save_synthetic_data(output_path=args.data)
        print("✓ Data generated")
    
    # Connect to Redis
    print(f"\n🔗 Connecting to Redis at {args.redis_host}:{args.redis_port}...")
    store = RedisFeatureStore(host=args.redis_host, port=args.redis_port)
    
    if not store.ping():
        print("❌ Cannot connect to Redis!")
        print("   Make sure Redis is running:")
        print("   docker run -d -p 6379:6379 redis:7-alpine")
        return
    
    print("✓ Connected to Redis")
    
    # Load data
    print("\n📂 Loading data from files...")
    loader = DataLoader(args.data)
    
    try:
        items = loader.load_items()
        print(f"  - Loaded {len(items)} items")
    except Exception as e:
        print(f"  ⚠ Could not load items: {e}")
        items = []
    
    try:
        users = loader.load_users()
        print(f"  - Loaded {len(users)} users")
    except Exception as e:
        print(f"  ⚠ Could not load users: {e}")
        users = []
    
    # Store items
    if items:
        print("\n📦 Storing items in Redis...")
        store.set_items_batch(items)
        print(f"✓ Stored {len(items)} items")
    
    # Store users
    if users:
        print("\n👤 Storing users in Redis...")
        for user in users:
            store.set_user(user)
        print(f"✓ Stored {len(users)} users")
    
    # Generate and store embeddings
    if args.embeddings and items:
        print("\n🧠 Generating embeddings...")
        try:
            from src.genai.embeddings import EmbeddingService
            
            embedding_service = EmbeddingService()
            embeddings = embedding_service.embed_items(items)
            
            print(f"  - Embedding dimension: {embeddings.shape[1]}")
            
            item_ids = [item.item_id for item in items]
            store.set_embeddings_batch(item_ids, embeddings)
            
            print(f"✓ Stored {len(items)} embeddings")
        except Exception as e:
            print(f"  ⚠ Could not generate embeddings: {e}")
    
    # Show stats
    print("\n📈 Redis Statistics:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Feature loading complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
