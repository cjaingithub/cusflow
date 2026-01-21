#!/usr/bin/env python3
"""
Run A/B test simulation between two ranking models.

Usage:
    python scripts/run_ab_sim.py --control models/baseline.joblib --treatment models/lambdamart_v1.joblib
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.config import Domain
from src.data.loaders import DataLoader, SyntheticDataGenerator
from src.evaluation.ab_simulation import ABSimulator, SimulationConfig
from src.ranking.lambdamart import LambdaMARTRanker


class SimpleRanker:
    """Simple baseline ranker for comparison."""
    
    def __init__(self, strategy: str = "popularity"):
        self.strategy = strategy
        self.item_scores: dict[str, float] = {}
    
    def fit(self, items: list) -> None:
        """Compute item scores based on strategy."""
        for item in items:
            if self.strategy == "popularity":
                self.item_scores[item.item_id] = item.features.popularity_score
            elif self.strategy == "quality":
                self.item_scores[item.item_id] = item.features.quality_score
            elif self.strategy == "random":
                import random
                self.item_scores[item.item_id] = random.random()
    
    def rank(self, user, candidate_ids, context=None) -> list[tuple[str, float]]:
        """Rank candidates."""
        scored = [
            (cid, self.item_scores.get(cid, 0))
            for cid in candidate_ids
        ]
        return sorted(scored, key=lambda x: -x[1])


class ModelRanker:
    """Wrapper for LambdaMART model."""
    
    def __init__(self, model_path: Path, items: list):
        self.model = LambdaMARTRanker()
        self.model.load(model_path)
        self.item_features = {item.item_id: item for item in items}
    
    def rank(self, user, candidate_ids, context=None) -> list[tuple[str, float]]:
        """Rank using the trained model."""
        import numpy as np
        from src.ranking.feature_engineering import FeatureEngineer
        
        items = [self.item_features[cid] for cid in candidate_ids if cid in self.item_features]
        if not items:
            return [(cid, 0) for cid in candidate_ids]
        
        fe = FeatureEngineer()
        try:
            fe.fit(items)
            X = fe.transform(items, user=user, context=context)
            return self.model.rank(X, [item.item_id for item in items])
        except Exception:
            return [(item.item_id, item.features.popularity_score) for item in items]


def main():
    parser = argparse.ArgumentParser(description="Run A/B simulation")
    parser.add_argument("--data", type=Path, default=Path("data/"))
    parser.add_argument("--control", type=str, default="popularity", help="Control: popularity, quality, random, or model path")
    parser.add_argument("--treatment", type=Path, default=Path("models/lambdamart_v1.joblib"))
    parser.add_argument("--output", type=Path, default=Path("reports/"))
    parser.add_argument("--traffic-ratio", type=float, default=0.5)
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data")
    args = parser.parse_args()
    
    print("=" * 60)
    print("A/B Test Simulation")
    print("=" * 60)
    
    # Generate data if needed
    if args.generate or not (args.data / "events.parquet").exists():
        print("\n📊 Generating synthetic data...")
        generator = SyntheticDataGenerator(domain=Domain.HOTEL)
        generator.save_synthetic_data(
            output_path=args.data,
            n_items=500,
            n_users=200,
            n_events=5000,
        )
    
    # Load data
    print("\n📂 Loading data...")
    loader = DataLoader(args.data)
    
    items = loader.load_items()
    users = loader.load_users()
    events = loader.load_events()
    
    print(f"  - Items: {len(items)}")
    print(f"  - Users: {len(users)}")
    print(f"  - Events: {len(events)}")
    
    # Create rankers
    print("\n🔧 Setting up rankers...")
    
    # Control ranker
    if args.control in ["popularity", "quality", "random"]:
        control_ranker = SimpleRanker(strategy=args.control)
        control_ranker.fit(items)
        print(f"  - Control: {args.control} baseline")
    else:
        control_ranker = ModelRanker(Path(args.control), items)
        print(f"  - Control: model from {args.control}")
    
    # Treatment ranker
    if args.treatment.exists():
        treatment_ranker = ModelRanker(args.treatment, items)
        print(f"  - Treatment: model from {args.treatment}")
    else:
        treatment_ranker = SimpleRanker(strategy="quality")
        treatment_ranker.fit(items)
        print("  - Treatment: quality baseline (model not found)")
    
    # Create user and item lookups
    user_dict = {u.user_id: u for u in users}
    item_dict = {i.item_id: i for i in items}
    
    # Run simulation
    print(f"\n🚀 Running simulation (traffic ratio: {args.traffic_ratio})...")
    
    config = SimulationConfig(
        experiment_id="ab_sim_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        treatment_ratio=args.traffic_ratio,
    )
    
    simulator = ABSimulator(
        control_ranker=control_ranker,
        treatment_ranker=treatment_ranker,
        config=config,
    )
    
    result = simulator.simulate(events, user_dict, item_dict)
    
    # Print report
    report = simulator.generate_report(result)
    print("\n" + report)
    
    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result_path = args.output / f"ab_simulation_{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump(result.model_dump(), f, indent=2, default=str)
    
    report_path = args.output / f"ab_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"\n💾 Results saved to {result_path}")
    print(f"📄 Report saved to {report_path}")
    
    print("\n" + "=" * 60)
    print("✅ Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
