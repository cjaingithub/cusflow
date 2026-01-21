"""
CusFlow CLI

Command-line interface for managing the recommendation system.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="cusflow",
    help="CusFlow - Production-Ready LTR Recommendation System",
    add_completion=False,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
) -> None:
    """Start the CusFlow API server."""
    import uvicorn
    
    console.print(f"[green]🚀 Starting CusFlow API on {host}:{port}[/green]")
    
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
    )


@app.command()
def generate_data(
    domain: str = typer.Option("hotel", "--domain", "-d", help="Domain: hotel, wealth_report, ecommerce"),
    n_items: int = typer.Option(1000, "--items", "-i", help="Number of items to generate"),
    n_users: int = typer.Option(500, "--users", "-u", help="Number of users to generate"),
    n_events: int = typer.Option(10000, "--events", "-e", help="Number of events to generate"),
    output: Path = typer.Option(Path("data/"), "--output", "-o", help="Output directory"),
) -> None:
    """Generate synthetic data for testing."""
    from src.config import Domain
    from src.data.loaders import SyntheticDataGenerator
    
    console.print(f"[blue]Generating synthetic data for domain: {domain}[/blue]")
    
    domain_enum = Domain(domain)
    generator = SyntheticDataGenerator(domain=domain_enum)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating...", total=None)
        
        paths = generator.save_synthetic_data(
            output_path=output,
            n_items=n_items,
            n_users=n_users,
            n_events=n_events,
        )
        
        progress.update(task, completed=True)
    
    console.print("[green]✓ Data generated successfully![/green]")
    
    table = Table(title="Generated Files")
    table.add_column("Type", style="cyan")
    table.add_column("Path", style="green")
    
    for name, path in paths.items():
        table.add_row(name, str(path))
    
    console.print(table)


@app.command()
def train(
    data_path: Path = typer.Option(Path("data/"), "--data", "-d", help="Data directory"),
    model_path: Path = typer.Option(Path("models/"), "--model", "-m", help="Model output directory"),
    model_type: str = typer.Option("lambdamart", "--type", "-t", help="Model type: lambdamart, xgboost"),
    n_estimators: int = typer.Option(500, "--estimators", "-n", help="Number of boosting rounds"),
) -> None:
    """Train the ranking model."""
    from src.data.loaders import DataLoader
    from src.ranking.lambdamart import LambdaMARTRanker, XGBoostRanker
    
    console.print("[blue]Loading training data...[/blue]")
    
    loader = DataLoader(data_path)
    X, y, groups = loader.load_training_data()
    
    console.print(f"[green]✓ Loaded {len(y)} examples in {len(groups)} queries[/green]")
    
    console.print(f"[blue]Training {model_type} model...[/blue]")
    
    if model_type == "lambdamart":
        model = LambdaMARTRanker(num_boost_round=n_estimators)
    else:
        model = XGBoostRanker(num_boost_round=n_estimators)
    
    # Split data
    n_train = int(len(groups) * 0.8)
    train_size = sum(groups[:n_train])
    
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    groups_train, groups_val = groups[:n_train], groups[n_train:]
    
    model.fit(
        X_train, y_train, groups_train,
        X_val=X_val, y_val=y_val, groups_val=groups_val,
    )
    
    # Save model
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(model_path / "lambdamart_v1.joblib")
    
    console.print(f"[green]✓ Model saved to {model_path}[/green]")
    
    # Show feature importance
    if hasattr(model, 'get_feature_importance'):
        importance = model.get_feature_importance(top_k=10)
        
        table = Table(title="Top 10 Feature Importance")
        table.add_column("Feature", style="cyan")
        table.add_column("Importance", style="green")
        
        for feature, score in importance.items():
            table.add_row(feature, f"{score:.4f}")
        
        console.print(table)


@app.command()
def evaluate(
    data_path: Path = typer.Option(Path("data/"), "--data", "-d", help="Data directory"),
    model_path: Path = typer.Option(Path("models/lambdamart_v1.joblib"), "--model", "-m", help="Model path"),
) -> None:
    """Evaluate the ranking model."""
    from src.data.loaders import DataLoader
    from src.evaluation.metrics import RankingMetrics
    from src.ranking.lambdamart import LambdaMARTRanker
    
    console.print("[blue]Loading model and data...[/blue]")
    
    loader = DataLoader(data_path)
    X, y, groups = loader.load_training_data()
    
    model = LambdaMARTRanker()
    model.load(model_path)
    
    console.print("[blue]Evaluating...[/blue]")
    
    y_pred = model.predict(X)
    
    metrics = RankingMetrics(cutoffs=[5, 10, 20])
    results = metrics.evaluate(y, y_pred, groups)
    
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for metric, value in sorted(results.items()):
        table.add_row(metric, f"{value:.4f}")
    
    console.print(table)


@app.command()
def ablation(
    data_path: Path = typer.Option(Path("data/"), "--data", "-d", help="Data directory"),
) -> None:
    """Run ablation study on features."""
    from src.data.loaders import DataLoader
    from src.evaluation.ablation import AblationStudy
    from src.ranking.lambdamart import LambdaMARTRanker
    
    console.print("[blue]Running ablation study...[/blue]")
    
    loader = DataLoader(data_path)
    X, y, groups = loader.load_training_data()
    
    # Get feature names from training data
    feature_names = [f"f_{i}" for i in range(X.shape[1])]
    
    study = AblationStudy(
        model_class=LambdaMARTRanker,
        model_params={"num_boost_round": 100},
    )
    
    result = study.run(X, y, groups, feature_names)
    
    report = study.generate_report(result)
    console.print(report)


@app.command()
def load_features(
    data_path: Path = typer.Option(Path("data/"), "--data", "-d", help="Data directory"),
    redis_host: str = typer.Option("localhost", "--redis-host", help="Redis host"),
    redis_port: int = typer.Option(6379, "--redis-port", help="Redis port"),
) -> None:
    """Load features into Redis."""
    from src.data.loaders import DataLoader
    from src.store.redis_store import RedisFeatureStore
    
    console.print("[blue]Loading features into Redis...[/blue]")
    
    loader = DataLoader(data_path)
    store = RedisFeatureStore(host=redis_host, port=redis_port)
    
    # Check Redis connection
    if not store.ping():
        console.print("[red]✗ Cannot connect to Redis[/red]")
        raise typer.Exit(1)
    
    # Load items
    try:
        items = loader.load_items()
        store.set_items_batch(items)
        console.print(f"[green]✓ Loaded {len(items)} items[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not load items: {e}[/yellow]")
    
    # Load users
    try:
        users = loader.load_users()
        for user in users:
            store.set_user(user)
        console.print(f"[green]✓ Loaded {len(users)} users[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not load users: {e}[/yellow]")
    
    # Show stats
    stats = store.get_stats()
    console.print(f"[blue]Redis stats: {stats}[/blue]")


@app.command()
def info() -> None:
    """Show system information."""
    from src.config import get_settings
    
    settings = get_settings()
    
    table = Table(title="CusFlow Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Domain", settings.domain.value)
    table.add_row("API Host", settings.api_host)
    table.add_row("API Port", str(settings.api_port))
    table.add_row("Redis", f"{settings.redis_host}:{settings.redis_port}")
    table.add_row("Embedding Provider", settings.embedding_provider.value)
    table.add_row("Model Path", str(settings.model_path))
    table.add_row("Data Path", str(settings.data_path))
    
    console.print(table)


if __name__ == "__main__":
    app()
