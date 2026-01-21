"""
Approximate Nearest Neighbor (ANN) retrieval for candidate generation.

Uses FAISS for efficient similarity search to retrieve candidate items
before the ranking stage.
"""

import json
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from src.config import get_settings
from src.data.schemas import Item, User


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    def embed_items(self, items: list[Item]) -> np.ndarray:
        """Generate embeddings for items."""
        ...
    
    def embed_query(self, user: User, query: str | None = None) -> np.ndarray:
        """Generate query embedding from user context."""
        ...


class ANNRetriever:
    """
    FAISS-based ANN retriever for candidate generation.
    
    Supports multiple index types:
    - Flat: Exact search (for small catalogs)
    - IVF: Inverted file index (for medium catalogs)
    - HNSW: Hierarchical navigable small world graphs (for large catalogs)
    """
    
    def __init__(
        self,
        dimension: int = 384,  # Default for all-MiniLM-L6-v2
        index_type: str = "IVF",
        nlist: int = 100,  # Number of clusters for IVF
        nprobe: int = 10,  # Number of clusters to search
        use_gpu: bool = False,
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.settings = get_settings()
        
        self.index: faiss.Index | None = None
        self.item_ids: list[str] = []
        self.is_trained = False
    
    def build_index(
        self, 
        embeddings: np.ndarray, 
        item_ids: list[str],
    ) -> None:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: (n_items, dimension) array of item embeddings
            item_ids: List of item IDs corresponding to embeddings
        """
        n_items, dim = embeddings.shape
        assert dim == self.dimension, f"Dimension mismatch: {dim} vs {self.dimension}"
        assert len(item_ids) == n_items, "Number of IDs must match embeddings"
        
        self.item_ids = item_ids
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        if self.index_type == "Flat":
            # Exact search - good for < 10k items
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product after L2 norm = cosine
            self.index.add(embeddings)
            self.is_trained = True
            
        elif self.index_type == "IVF":
            # IVF index - good for 10k-1M items
            quantizer = faiss.IndexFlatIP(self.dimension)
            nlist = min(self.nlist, n_items // 10)  # Ensure reasonable cluster count
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Train on data
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = self.nprobe
            self.is_trained = True
            
        elif self.index_type == "HNSW":
            # HNSW - good for > 1M items
            self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.add(embeddings)
            self.is_trained = True
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Move to GPU if available and requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self.index
            )
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 100,
    ) -> tuple[list[str], list[float]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_embedding: (1, dimension) or (dimension,) query vector
            k: Number of neighbors to retrieve
            
        Returns:
            item_ids: List of retrieved item IDs
            scores: List of similarity scores
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index first.")
        
        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(k, len(self.item_ids))
        scores, indices = self.index.search(query_embedding, k)
        
        # Map indices to item IDs
        retrieved_ids = [self.item_ids[idx] for idx in indices[0] if idx >= 0]
        retrieved_scores = [float(score) for score, idx in zip(scores[0], indices[0]) if idx >= 0]
        
        return retrieved_ids, retrieved_scores
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 100,
    ) -> list[tuple[list[str], list[float]]]:
        """
        Batch search for multiple queries.
        
        Args:
            query_embeddings: (n_queries, dimension) query vectors
            k: Number of neighbors per query
            
        Returns:
            List of (item_ids, scores) tuples
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index first.")
        
        # Normalize queries
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)
        
        k = min(k, len(self.item_ids))
        scores, indices = self.index.search(query_embeddings, k)
        
        results = []
        for i in range(len(query_embeddings)):
            ids = [self.item_ids[idx] for idx in indices[i] if idx >= 0]
            scs = [float(score) for score, idx in zip(scores[i], indices[i]) if idx >= 0]
            results.append((ids, scs))
        
        return results
    
    def save(self, path: Path | None = None) -> None:
        """Save index to disk."""
        if self.index is None:
            raise RuntimeError("No index to save")
        
        path = path or self.settings.candidate_model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path))
        
        # Save metadata
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump({
                "dimension": self.dimension,
                "index_type": self.index_type,
                "item_ids": self.item_ids,
                "nlist": self.nlist,
                "nprobe": self.nprobe,
            }, f)
    
    def load(self, path: Path | None = None) -> None:
        """Load index from disk."""
        path = path or self.settings.candidate_model_path
        
        if not path.exists():
            raise FileNotFoundError(f"Index not found at {path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(path))
        
        # Load metadata
        meta_path = path.with_suffix(".json")
        with open(meta_path) as f:
            meta = json.load(f)
        
        self.dimension = meta["dimension"]
        self.index_type = meta["index_type"]
        self.item_ids = meta["item_ids"]
        self.nlist = meta.get("nlist", 100)
        self.nprobe = meta.get("nprobe", 10)
        self.is_trained = True


class CandidateGenerator:
    """
    High-level candidate generation combining multiple retrieval strategies.
    
    Strategies:
    1. ANN search on item embeddings
    2. Popular items (fallback)
    3. User history-based (collaborative filtering)
    """
    
    def __init__(
        self,
        ann_retriever: ANNRetriever | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.settings = get_settings()
        self.ann_retriever = ann_retriever or ANNRetriever()
        self.embedding_provider = embedding_provider
        
        # Popular items fallback
        self.popular_items: list[str] = []
    
    def set_popular_items(self, item_ids: list[str]) -> None:
        """Set fallback popular items list."""
        self.popular_items = item_ids
    
    def generate_candidates(
        self,
        user: User,
        query: str | None = None,
        k: int = 100,
        strategies: list[str] | None = None,
    ) -> tuple[list[str], dict[str, list[str]]]:
        """
        Generate candidate items for a user.
        
        Args:
            user: User requesting recommendations
            query: Optional search query
            k: Number of candidates to retrieve
            strategies: Which strategies to use (default: all)
            
        Returns:
            candidate_ids: Deduplicated list of candidate item IDs
            strategy_results: Results from each strategy
        """
        strategies = strategies or ["ann", "popular"]
        strategy_results: dict[str, list[str]] = {}
        
        if "ann" in strategies and self.embedding_provider and self.ann_retriever.is_trained:
            # Get user/query embedding
            query_emb = self.embedding_provider.embed_query(user, query)
            ids, _ = self.ann_retriever.search(query_emb, k=k)
            strategy_results["ann"] = ids
        
        if "popular" in strategies and self.popular_items:
            strategy_results["popular"] = self.popular_items[:k]
        
        # Merge and deduplicate (preserving order from primary strategy)
        seen = set()
        candidates = []
        
        for strategy in strategies:
            for item_id in strategy_results.get(strategy, []):
                if item_id not in seen:
                    seen.add(item_id)
                    candidates.append(item_id)
        
        return candidates[:k], strategy_results
    
    def build_from_items(
        self,
        items: list[Item],
        embeddings: np.ndarray | None = None,
    ) -> None:
        """
        Build candidate generation index from items.
        
        Args:
            items: List of items
            embeddings: Pre-computed embeddings (optional)
        """
        item_ids = [item.item_id for item in items]
        
        if embeddings is None:
            if self.embedding_provider is None:
                raise ValueError("Need embeddings or embedding_provider")
            embeddings = self.embedding_provider.embed_items(items)
        
        # Update retriever dimension based on embeddings
        self.ann_retriever.dimension = embeddings.shape[1]
        
        # Build index
        self.ann_retriever.build_index(embeddings, item_ids)
        
        # Set popular items based on popularity score
        sorted_items = sorted(
            items, 
            key=lambda x: x.features.popularity_score, 
            reverse=True
        )
        self.popular_items = [item.item_id for item in sorted_items]
    
    def save(self, path: Path | None = None) -> None:
        """Save candidate generator to disk."""
        self.ann_retriever.save(path)
    
    def load(self, path: Path | None = None) -> None:
        """Load candidate generator from disk."""
        self.ann_retriever.load(path)
