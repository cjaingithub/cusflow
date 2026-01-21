"""
Redis-based feature store for real-time feature serving.

Provides fast read/write access to:
- Item features
- User features  
- Pre-computed embeddings
- Real-time aggregations
"""

import json
from datetime import timedelta
from typing import Any

import numpy as np
import redis
from redis.asyncio import Redis as AsyncRedis

from src.config import get_settings
from src.data.schemas import Item, ItemFeatures, User, UserFeatures


class RedisFeatureStore:
    """
    Redis-based feature store for serving features at inference time.
    
    Features:
    - Fast key-value lookups for item/user features
    - Support for batch operations
    - TTL-based expiration for freshness
    - Async support for FastAPI integration
    """
    
    # Key prefixes
    ITEM_PREFIX = "item:"
    USER_PREFIX = "user:"
    EMBEDDING_PREFIX = "emb:"
    FEATURE_PREFIX = "feat:"
    
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        password: str | None = None,
        db: int | None = None,
        ttl_seconds: int | None = None,
    ):
        self.settings = get_settings()
        
        self.host = host or self.settings.redis_host
        self.port = port or self.settings.redis_port
        self.password = password or self.settings.redis_password or None
        self.db = db if db is not None else self.settings.redis_db
        self.ttl = ttl_seconds or self.settings.redis_ttl_seconds
        
        self._client: redis.Redis | None = None
        self._async_client: AsyncRedis | None = None
    
    @property
    def client(self) -> redis.Redis:
        """Get synchronous Redis client."""
        if self._client is None:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=True,
            )
        return self._client
    
    @property
    def async_client(self) -> AsyncRedis:
        """Get async Redis client."""
        if self._async_client is None:
            self._async_client = AsyncRedis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=True,
            )
        return self._async_client
    
    def close(self) -> None:
        """Close connections."""
        if self._client:
            self._client.close()
        # Async client close needs to be awaited
    
    async def aclose(self) -> None:
        """Async close connections."""
        if self._async_client:
            await self._async_client.close()
    
    # =========================================================================
    # Item Operations
    # =========================================================================
    
    def set_item(self, item: Item) -> None:
        """Store item features in Redis."""
        key = f"{self.ITEM_PREFIX}{item.item_id}"
        
        data = {
            "item_id": item.item_id,
            "domain": item.domain,
            "name": item.name,
            "features": json.dumps(item.features.features),
            "quality_score": item.features.quality_score,
            "popularity_score": item.features.popularity_score,
        }
        
        if item.features.summary:
            data["summary"] = item.features.summary
        
        self.client.hset(key, mapping=data)
        self.client.expire(key, self.ttl)
    
    def get_item(self, item_id: str) -> Item | None:
        """Retrieve item from Redis."""
        key = f"{self.ITEM_PREFIX}{item_id}"
        data = self.client.hgetall(key)
        
        if not data:
            return None
        
        features = ItemFeatures(
            features=json.loads(data.get("features", "{}")),
            quality_score=float(data.get("quality_score", 0)),
            popularity_score=float(data.get("popularity_score", 0)),
            summary=data.get("summary"),
        )
        
        return Item(
            item_id=data["item_id"],
            domain=data.get("domain", "unknown"),
            name=data.get("name", ""),
            features=features,
        )
    
    def set_items_batch(self, items: list[Item]) -> None:
        """Store multiple items in a pipeline."""
        pipe = self.client.pipeline()
        
        for item in items:
            key = f"{self.ITEM_PREFIX}{item.item_id}"
            data = {
                "item_id": item.item_id,
                "domain": item.domain,
                "name": item.name,
                "features": json.dumps(item.features.features),
                "quality_score": item.features.quality_score,
                "popularity_score": item.features.popularity_score,
            }
            if item.features.summary:
                data["summary"] = item.features.summary
            
            pipe.hset(key, mapping=data)
            pipe.expire(key, self.ttl)
        
        pipe.execute()
    
    def get_items_batch(self, item_ids: list[str]) -> list[Item | None]:
        """Retrieve multiple items in a pipeline."""
        pipe = self.client.pipeline()
        
        for item_id in item_ids:
            key = f"{self.ITEM_PREFIX}{item_id}"
            pipe.hgetall(key)
        
        results = pipe.execute()
        items = []
        
        for data in results:
            if not data:
                items.append(None)
                continue
            
            features = ItemFeatures(
                features=json.loads(data.get("features", "{}")),
                quality_score=float(data.get("quality_score", 0)),
                popularity_score=float(data.get("popularity_score", 0)),
                summary=data.get("summary"),
            )
            
            items.append(Item(
                item_id=data["item_id"],
                domain=data.get("domain", "unknown"),
                name=data.get("name", ""),
                features=features,
            ))
        
        return items
    
    # =========================================================================
    # User Operations
    # =========================================================================
    
    def set_user(self, user: User) -> None:
        """Store user features in Redis."""
        key = f"{self.USER_PREFIX}{user.user_id}"
        
        data = {
            "user_id": user.user_id,
            "features": json.dumps(self._serialize_features(user.features.features)),
            "total_interactions": user.features.total_interactions,
            "segments": json.dumps(user.segments),
        }
        
        self.client.hset(key, mapping=data)
        self.client.expire(key, self.ttl)
    
    def get_user(self, user_id: str) -> User | None:
        """Retrieve user from Redis."""
        key = f"{self.USER_PREFIX}{user_id}"
        data = self.client.hgetall(key)
        
        if not data:
            return None
        
        features = UserFeatures(
            features=json.loads(data.get("features", "{}")),
            total_interactions=int(data.get("total_interactions", 0)),
        )
        
        return User(
            user_id=data["user_id"],
            features=features,
            segments=json.loads(data.get("segments", "[]")),
        )
    
    # =========================================================================
    # Embedding Operations
    # =========================================================================
    
    def set_embedding(
        self, 
        entity_id: str, 
        embedding: np.ndarray,
        entity_type: str = "item",
    ) -> None:
        """Store embedding vector."""
        key = f"{self.EMBEDDING_PREFIX}{entity_type}:{entity_id}"
        
        # Store as JSON list for simplicity
        # For production, consider using Redis' native vector support
        self.client.set(
            key,
            json.dumps(embedding.tolist()),
            ex=self.ttl,
        )
    
    def get_embedding(
        self, 
        entity_id: str,
        entity_type: str = "item",
    ) -> np.ndarray | None:
        """Retrieve embedding vector."""
        key = f"{self.EMBEDDING_PREFIX}{entity_type}:{entity_id}"
        data = self.client.get(key)
        
        if data is None:
            return None
        
        return np.array(json.loads(data), dtype=np.float32)
    
    def set_embeddings_batch(
        self,
        entity_ids: list[str],
        embeddings: np.ndarray,
        entity_type: str = "item",
    ) -> None:
        """Store multiple embeddings in a pipeline."""
        pipe = self.client.pipeline()
        
        for entity_id, embedding in zip(entity_ids, embeddings):
            key = f"{self.EMBEDDING_PREFIX}{entity_type}:{entity_id}"
            pipe.set(key, json.dumps(embedding.tolist()), ex=self.ttl)
        
        pipe.execute()
    
    def get_embeddings_batch(
        self,
        entity_ids: list[str],
        entity_type: str = "item",
    ) -> list[np.ndarray | None]:
        """Retrieve multiple embeddings in a pipeline."""
        pipe = self.client.pipeline()
        
        for entity_id in entity_ids:
            key = f"{self.EMBEDDING_PREFIX}{entity_type}:{entity_id}"
            pipe.get(key)
        
        results = pipe.execute()
        
        embeddings = []
        for data in results:
            if data is None:
                embeddings.append(None)
            else:
                embeddings.append(np.array(json.loads(data), dtype=np.float32))
        
        return embeddings
    
    # =========================================================================
    # Async Operations
    # =========================================================================
    
    async def aget_item(self, item_id: str) -> Item | None:
        """Async retrieve item from Redis."""
        key = f"{self.ITEM_PREFIX}{item_id}"
        data = await self.async_client.hgetall(key)
        
        if not data:
            return None
        
        features = ItemFeatures(
            features=json.loads(data.get("features", "{}")),
            quality_score=float(data.get("quality_score", 0)),
            popularity_score=float(data.get("popularity_score", 0)),
            summary=data.get("summary"),
        )
        
        return Item(
            item_id=data["item_id"],
            domain=data.get("domain", "unknown"),
            name=data.get("name", ""),
            features=features,
        )
    
    async def aget_items_batch(self, item_ids: list[str]) -> list[Item | None]:
        """Async retrieve multiple items."""
        pipe = self.async_client.pipeline()
        
        for item_id in item_ids:
            key = f"{self.ITEM_PREFIX}{item_id}"
            pipe.hgetall(key)
        
        results = await pipe.execute()
        items = []
        
        for data in results:
            if not data:
                items.append(None)
                continue
            
            features = ItemFeatures(
                features=json.loads(data.get("features", "{}")),
                quality_score=float(data.get("quality_score", 0)),
                popularity_score=float(data.get("popularity_score", 0)),
                summary=data.get("summary"),
            )
            
            items.append(Item(
                item_id=data["item_id"],
                domain=data.get("domain", "unknown"),
                name=data.get("name", ""),
                features=features,
            ))
        
        return items
    
    async def aget_user(self, user_id: str) -> User | None:
        """Async retrieve user from Redis."""
        key = f"{self.USER_PREFIX}{user_id}"
        data = await self.async_client.hgetall(key)
        
        if not data:
            return None
        
        features = UserFeatures(
            features=json.loads(data.get("features", "{}")),
            total_interactions=int(data.get("total_interactions", 0)),
        )
        
        return User(
            user_id=data["user_id"],
            features=features,
            segments=json.loads(data.get("segments", "[]")),
        )
    
    async def aget_embedding(
        self,
        entity_id: str,
        entity_type: str = "item",
    ) -> np.ndarray | None:
        """Async retrieve embedding."""
        key = f"{self.EMBEDDING_PREFIX}{entity_type}:{entity_id}"
        data = await self.async_client.get(key)
        
        if data is None:
            return None
        
        return np.array(json.loads(data), dtype=np.float32)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _serialize_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Serialize features dict for JSON storage."""
        result = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                result[key] = value
            else:
                result[key] = str(value)
        return result
    
    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return self.client.ping()
        except redis.ConnectionError:
            return False
    
    async def aping(self) -> bool:
        """Async check Redis connectivity."""
        try:
            return await self.async_client.ping()
        except redis.ConnectionError:
            return False
    
    def flush_all(self) -> None:
        """Clear all keys (use with caution!)."""
        self.client.flushdb()
    
    def get_stats(self) -> dict[str, int]:
        """Get store statistics."""
        info = self.client.info()
        
        # Count keys by prefix
        item_count = len(list(self.client.scan_iter(f"{self.ITEM_PREFIX}*")))
        user_count = len(list(self.client.scan_iter(f"{self.USER_PREFIX}*")))
        embedding_count = len(list(self.client.scan_iter(f"{self.EMBEDDING_PREFIX}*")))
        
        return {
            "total_keys": info.get("db0", {}).get("keys", 0) if isinstance(info.get("db0"), dict) else 0,
            "item_count": item_count,
            "user_count": user_count,
            "embedding_count": embedding_count,
            "used_memory_mb": info.get("used_memory", 0) // (1024 * 1024),
        }
