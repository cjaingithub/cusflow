"""
Embedding generation for items and queries.

Supports multiple providers:
- Local: sentence-transformers (free, runs locally)
- OpenAI: text-embedding-3-small/large (API-based)
"""

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from tqdm import tqdm

from src.config import EmbeddingProvider, get_settings
from src.data.schemas import Item, User


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...
    
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            (n_texts, dimension) numpy array
        """
        ...
    
    def embed_items(self, items: list[Item]) -> np.ndarray:
        """Generate embeddings for items using their text content."""
        texts = [item.get_text_for_embedding() for item in items]
        return self.embed_texts(texts)
    
    def embed_query(self, user: User, query: str | None = None) -> np.ndarray:
        """
        Generate query embedding from user context.
        
        For now, uses the query text. Can be extended to use user preferences.
        """
        if query:
            return self.embed_texts([query])[0]
        
        # Fallback: create embedding from user preferences
        pref_text = self._user_to_text(user)
        return self.embed_texts([pref_text])[0]
    
    def _user_to_text(self, user: User) -> str:
        """Convert user features to text for embedding."""
        parts = [f"user preferences:"]
        for key, value in user.features.features.items():
            if not isinstance(value, (list, dict)):
                parts.append(f"{key}={value}")
        return " ".join(parts)


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.
    
    Runs entirely locally, no API costs.
    """
    
    # Model dimensions
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "multi-qa-MiniLM-L6-cos-v1": 384,
        "all-distilroberta-v1": 768,
    }
    
    def __init__(self, model_name: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer
        
        self.settings = get_settings()
        self.model_name = model_name or self.settings.local_embedding_model
        self.batch_size = batch_size
        
        self._model = SentenceTransformer(self.model_name)
        self._dimension = self.MODEL_DIMENSIONS.get(
            self.model_name, 
            self._model.get_sentence_embedding_dimension()
        )
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers."""
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI embedding provider using text-embedding-3 models.
    
    Requires OPENAI_API_KEY environment variable.
    """
    
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self, 
        model_name: str | None = None,
        batch_size: int = 100,
        dimensions: int | None = None,  # For dimension reduction
    ):
        import openai
        
        self.settings = get_settings()
        self.model_name = model_name or self.settings.openai_embedding_model
        self.batch_size = batch_size
        self.requested_dimensions = dimensions
        
        self._client = openai.OpenAI(api_key=self.settings.openai_api_key)
        
        # Set dimension
        if dimensions:
            self._dimension = dimensions
        else:
            self._dimension = self.MODEL_DIMENSIONS.get(self.model_name, 1536)
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding"):
            batch = texts[i:i + self.batch_size]
            
            kwargs = {"model": self.model_name, "input": batch}
            if self.requested_dimensions and "text-embedding-3" in self.model_name:
                kwargs["dimensions"] = self.requested_dimensions
            
            response = self._client.embeddings.create(**kwargs)
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)


class EmbeddingService:
    """
    High-level embedding service that manages providers and caching.
    """
    
    def __init__(self, provider: BaseEmbeddingProvider | None = None):
        self.settings = get_settings()
        
        if provider:
            self._provider = provider
        elif self.settings.embedding_provider == EmbeddingProvider.LOCAL:
            self._provider = LocalEmbeddingProvider()
        elif self.settings.embedding_provider == EmbeddingProvider.OPENAI:
            self._provider = OpenAIEmbeddingProvider()
        else:
            # Default to local
            self._provider = LocalEmbeddingProvider()
        
        self._cache: dict[str, np.ndarray] = {}
    
    @property
    def dimension(self) -> int:
        return self._provider.dimension
    
    def embed_items(
        self, 
        items: list[Item],
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for items.
        
        Args:
            items: List of items to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            (n_items, dimension) numpy array
        """
        if not use_cache:
            return self._provider.embed_items(items)
        
        # Check cache and identify items to embed
        embeddings = []
        items_to_embed = []
        indices_to_embed = []
        
        for i, item in enumerate(items):
            if item.item_id in self._cache:
                embeddings.append((i, self._cache[item.item_id]))
            else:
                items_to_embed.append(item)
                indices_to_embed.append(i)
        
        # Embed uncached items
        if items_to_embed:
            new_embeddings = self._provider.embed_items(items_to_embed)
            
            # Update cache
            for item, emb in zip(items_to_embed, new_embeddings):
                self._cache[item.item_id] = emb
            
            # Add to results
            for idx, emb in zip(indices_to_embed, new_embeddings):
                embeddings.append((idx, emb))
        
        # Sort by original index and stack
        embeddings.sort(key=lambda x: x[0])
        return np.stack([emb for _, emb in embeddings])
    
    def embed_query(self, user: User, query: str | None = None) -> np.ndarray:
        """Generate query embedding."""
        return self._provider.embed_query(user, query)
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        item_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and items.
        
        Args:
            query_embedding: (dimension,) query vector
            item_embeddings: (n_items, dimension) item matrix
            
        Returns:
            (n_items,) similarity scores
        """
        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        item_norms = item_embeddings / (np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity
        return np.dot(item_norms, query_norm)
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()
