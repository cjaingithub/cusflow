#!/usr/bin/env python
"""
CusFlow: Expedia-Style Recommendation System Demo

Demonstrates:
1. Expedia-like hotel data generation
2. Dynamic pricing optimization
3. Multi-armed bandit personalization
4. Hybrid search (BM25 + semantic)
5. Learning-to-Rank with LambdaMART

Run: python scripts/demo_expedia.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def demo_pricing():
    """Demo dynamic pricing."""
    console.print("\n[bold cyan]💰 Dynamic Pricing Demo[/bold cyan]\n")
    
    from src.pricing.dynamic_pricing import DynamicPricingEngine, PricingContext
    engine = DynamicPricingEngine()
    
    scenarios = [
        ("High Demand + Low Inventory", PricingContext("h1", 150, 0.85, 0.1, 3, is_weekend=True)),
        ("Low Demand + High Inventory", PricingContext("h2", 150, 0.2, 0.8, 45)),
        ("Holiday Period", PricingContext("h3", 150, 0.7, 0.4, 14, is_holiday=True)),
    ]
    
    table = Table(title="Pricing Recommendations")
    table.add_column("Scenario"); table.add_column("Base"); table.add_column("Recommended"); table.add_column("Multiplier")
    
    for name, ctx in scenarios:
        rec = engine.optimize_price(ctx)
        table.add_row(name, f"${rec.base_price:.0f}", f"${rec.recommended_price:.0f}", f"{rec.price_multiplier:.2f}x")
    
    console.print(table)


def demo_bandits():
    """Demo multi-armed bandits."""
    console.print("\n[bold cyan]🎰 Multi-Armed Bandit Demo[/bold cyan]\n")
    
    from src.personalization.bandits import ThompsonSampling
    
    bandit = ThompsonSampling(["collaborative", "content_based", "popularity"])
    true_rates = {"collaborative": 0.15, "content_based": 0.12, "popularity": 0.08}
    
    for _ in range(500):
        result = bandit.select_arm()
        reward = 1 if np.random.random() < true_rates[result.selected_arm] else 0
        bandit.update(result.selected_arm, reward)
    
    stats = bandit.get_arm_stats()
    table = Table(title="Thompson Sampling Results (500 rounds)")
    table.add_column("Strategy"); table.add_column("True Rate"); table.add_column("Pulls"); table.add_column("Estimated")
    
    for s in true_rates:
        table.add_row(s, f"{true_rates[s]:.1%}", str(stats[s]["pulls"]), f"{stats[s]['mean_reward']:.1%}")
    
    console.print(table)


def demo_search():
    """Demo hybrid search."""
    console.print("\n[bold cyan]🔍 Hybrid Search Demo[/bold cyan]\n")
    
    from src.search.relevance import HybridSearch
    
    class Hotel:
        def __init__(self, id, desc):
            self.item_id = id
            self.description = desc
    
    hotels = [
        Hotel("h1", "Luxury beachfront resort with pool and spa"),
        Hotel("h2", "Budget downtown hotel with free wifi"),
        Hotel("h3", "Family resort with kids club and water park"),
        Hotel("h4", "Boutique hotel in historic district, pet friendly"),
    ]
    
    hybrid = HybridSearch(lexical_weight=0.4, semantic_weight=0.6)
    hybrid.index(hotels)
    
    response = hybrid.search("beach hotel pool", top_k=3)
    
    console.print(f"Query: 'beach hotel pool'\n")
    for r in response.results:
        console.print(f"  {r.position}. {r.item_id}: {r.score:.3f} (lex={r.lexical_score:.2f}, sem={r.semantic_score:.2f})")


def demo_data():
    """Demo Expedia data generation."""
    console.print("\n[bold cyan]📊 Expedia Data Generation Demo[/bold cyan]\n")
    
    from src.data.expedia_generator import ExpediaDataGenerator
    
    gen = ExpediaDataGenerator(seed=42)
    hotels = gen.generate_hotels(50)
    users = gen.generate_users(100)
    events = gen.generate_search_sessions(users, hotels, 200)
    
    console.print(f"Generated: {len(hotels)} hotels, {len(users)} users, {len(events)} events")
    
    h = hotels[0]
    console.print(f"\nSample Hotel: {h.name}")
    for k in ["star_rating", "review_score", "price_per_night", "chain"]:
        console.print(f"  {k}: {h.features.features[k]}")


def main():
    console.print(Panel.fit(
        "[bold magenta]CusFlow: Production Recommendation System Demo[/bold magenta]\n\n"
        "Demonstrating ML features for Expedia-style recommendations",
        title="🏨 CusFlow Demo", border_style="cyan"))
    
    try:
        demo_data()
        demo_pricing()
        demo_bandits()
        demo_search()
        
        console.print(Panel("[bold green]✅ All demos completed![/bold green]", border_style="green"))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
