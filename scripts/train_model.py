#!/usr/bin/env python3
"""
Train the LambdaMART ranking model.

Usage:
    python scripts/train_model.py --domain hotel --data data/ --output models/
"""

import argparse
from pathlib import Path

import numpy as np

from src.config import Domain, get_settings
from src.data.loaders import DataLoader, SyntheticDataGenerator
from src.evaluation.metrics import RankingMetrics
from src.ranking.bias_correction import InversePropensityWeighting
from src.ranking.lambdamart import LambdaMARTRanker


def main():
    parser = argparse.ArgumentParser(description="Train LambdaMART ranking model")
    parser.add_argument("--domain", type=str, default="hotel", choices=["hotel", "wealth_report", "ecommerce"])
    parser.add_argument("--data", type=Path, default=Path("data/"))
    parser.add_argument("--output", type=Path, default=Path("models/"))
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data first")
    parser.add_argument("--n-items", type=int, default=1000)
    parser.add_argument("--n-users", type=int, default=500)
    parser.add_argument("--n-events", type=int, default=10000)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--bias-correction", action="store_true", help="Apply click bias correction")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CusFlow Model Training")
    print("=" * 60)
    
    # Generate data if requested
    if args.generate or not (args.data / "training.parquet").exists():
        print(f"\n📊 Generating synthetic data for domain: {args.domain}")
        generator = SyntheticDataGenerator(domain=Domain(args.domain))
        generator.save_synthetic_data(
            output_path=args.data,
            n_items=args.n_items,
            n_users=args.n_users,
            n_events=args.n_events,
        )
        print("✓ Data generated")
    
    # Load training data
    print("\n📂 Loading training data...")
    loader = DataLoader(args.data)
    X, y, groups = loader.load_training_data()
    
    print(f"  - Samples: {len(y)}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Queries: {len(groups)}")
    
    # Apply bias correction if requested
    sample_weights = None
    if args.bias_correction:
        print("\n🔧 Applying click bias correction...")
        
        # Load events to get positions
        events = loader.load_events()
        positions = np.array([e.position or 1 for e in events])[:len(y)]
        clicks = (y > 0).astype(float)
        
        ipw = InversePropensityWeighting()
        ipw.fit(positions, clicks)
        sample_weights = ipw.compute_weights(positions, clicks)
        print("✓ Inverse propensity weights computed")
    
    # Split data
    print("\n🔀 Splitting data...")
    n_train = int(len(groups) * 0.8)
    train_size = sum(groups[:n_train])
    
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    groups_train, groups_val = groups[:n_train], groups[n_train:]
    
    if sample_weights is not None:
        sw_train = sample_weights[:train_size]
    else:
        sw_train = None
    
    print(f"  - Training: {train_size} samples, {n_train} queries")
    print(f"  - Validation: {len(y_val)} samples, {len(groups_val)} queries")
    
    # Train model
    print(f"\n🚀 Training LambdaMART model ({args.estimators} rounds)...")
    model = LambdaMARTRanker(num_boost_round=args.estimators)
    
    model.fit(
        X_train, y_train, groups_train,
        sample_weight=sw_train,
        X_val=X_val, y_val=y_val, groups_val=groups_val,
    )
    
    # Evaluate
    print("\n📈 Evaluating model...")
    y_pred = model.predict(X_val)
    
    metrics = RankingMetrics()
    results = metrics.evaluate(y_val, y_pred, groups_val)
    
    print("\n" + "-" * 40)
    print("Validation Metrics:")
    for metric, value in sorted(results.items()):
        print(f"  {metric}: {value:.4f}")
    print("-" * 40)
    
    # Feature importance
    print("\n🎯 Top 10 Feature Importance:")
    importance = model.get_feature_importance(top_k=10)
    for i, (feature, score) in enumerate(importance.items(), 1):
        print(f"  {i}. {feature}: {score:.4f}")
    
    # Save model
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "lambdamart_v1.joblib"
    model.save(model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
