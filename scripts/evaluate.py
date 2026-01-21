#!/usr/bin/env python3
"""
Evaluate the ranking model with comprehensive metrics.

Usage:
    python scripts/evaluate.py --model models/lambdamart_v1.joblib --data data/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data.loaders import DataLoader
from src.evaluation.ablation import AblationStudy
from src.evaluation.metrics import RankingMetrics
from src.ranking.lambdamart import LambdaMARTRanker


def main():
    parser = argparse.ArgumentParser(description="Evaluate ranking model")
    parser.add_argument("--model", type=Path, default=Path("models/lambdamart_v1.joblib"))
    parser.add_argument("--data", type=Path, default=Path("data/"))
    parser.add_argument("--output", type=Path, default=Path("reports/"))
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--bootstrap", action="store_true", help="Compute confidence intervals")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CusFlow Model Evaluation")
    print("=" * 60)
    
    # Load model
    print("\n📦 Loading model...")
    model = LambdaMARTRanker()
    model.load(args.model)
    print(f"✓ Model loaded from {args.model}")
    
    # Load data
    print("\n📂 Loading data...")
    loader = DataLoader(args.data)
    X, y, groups = loader.load_training_data()
    print(f"  - Samples: {len(y)}")
    print(f"  - Queries: {len(groups)}")
    
    # Predict
    print("\n🔮 Running predictions...")
    y_pred = model.predict(X)
    
    # Evaluate
    print("\n📊 Computing metrics...")
    metrics = RankingMetrics(cutoffs=[5, 10, 20, 50])
    
    if args.bootstrap:
        print("  (with bootstrap confidence intervals)")
        results = metrics.evaluate_with_confidence(y, y_pred, groups)
        
        print("\n" + "=" * 60)
        print("Metrics with 95% Confidence Intervals")
        print("=" * 60)
        
        for metric, (mean, lower, upper) in sorted(results.items()):
            print(f"  {metric:15s}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")
    else:
        results = metrics.evaluate(y, y_pred, groups)
        
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        
        for metric, value in sorted(results.items()):
            print(f"  {metric:15s}: {value:.4f}")
    
    # Ablation study
    if args.ablation:
        print("\n" + "=" * 60)
        print("Ablation Study")
        print("=" * 60)
        
        feature_names = [f"f_{i}" for i in range(X.shape[1])]
        
        study = AblationStudy(
            model_class=LambdaMARTRanker,
            model_params={"num_boost_round": 100},
        )
        
        ablation_result = study.run(X, y, groups, feature_names)
        report = study.generate_report(ablation_result)
        print(report)
    
    # Save report
    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.output / f"evaluation_{timestamp}.json"
    
    report_data = {
        "timestamp": timestamp,
        "model_path": str(args.model),
        "data_path": str(args.data),
        "n_samples": len(y),
        "n_queries": len(groups),
        "metrics": results if not args.bootstrap else {
            k: {"mean": v[0], "ci_lower": v[1], "ci_upper": v[2]}
            for k, v in results.items()
        },
    }
    
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n💾 Report saved to {report_path}")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
