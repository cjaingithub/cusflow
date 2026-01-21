"""
FastAPI Application Factory

Creates and configures the FastAPI application with all routes,
middleware, and dependencies.
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown."""
    # Startup
    settings = get_settings()
    print(f"🚀 Starting CusFlow API on {settings.api_host}:{settings.api_port}")
    
    # Initialize services
    app.state.settings = settings
    
    # Initialize Redis connection (lazy)
    from src.store.redis_store import RedisFeatureStore
    app.state.feature_store = RedisFeatureStore()
    
    # Load models if available
    try:
        from src.ranking.lambdamart import LambdaMARTRanker
        ranker = LambdaMARTRanker()
        if settings.ranking_model_path.exists():
            ranker.load()
            app.state.ranker = ranker
            print("✓ Ranking model loaded")
        else:
            app.state.ranker = None
            print("⚠ No ranking model found")
    except Exception as e:
        app.state.ranker = None
        print(f"⚠ Could not load ranking model: {e}")
    
    # Initialize embedding service
    try:
        from src.genai.embeddings import EmbeddingService
        app.state.embedding_service = EmbeddingService()
        print("✓ Embedding service initialized")
    except Exception as e:
        app.state.embedding_service = None
        print(f"⚠ Could not initialize embedding service: {e}")
    
    yield
    
    # Shutdown
    print("👋 Shutting down CusFlow API")
    
    # Close Redis connection
    if hasattr(app.state, "feature_store"):
        await app.state.feature_store.aclose()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="CusFlow Ranking API",
        description="""
        Production-ready Learning-to-Rank recommendation system with GenAI features.
        
        ## Features
        - 🎯 Personalized ranking with LambdaMART
        - 🤖 GenAI-powered item embeddings and summaries
        - ⚡ Real-time feature serving with Redis
        - 📊 Comprehensive metrics and A/B testing
        
        ## Domains Supported
        - Hotels (Expedia-style)
        - Wealth Management Reports
        - E-commerce Products
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        return response
    
    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.api_debug else "An error occurred",
            },
        )
    
    # Register routes
    from src.api.routes import router
    app.include_router(router)
    
    return app


# Create default app instance
app = create_app()
