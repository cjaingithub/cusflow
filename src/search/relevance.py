"""
Search Relevance Model

Implements hybrid search combining:
- BM25 lexical matching
- Semantic similarity (embeddings)
- Query understanding
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SearchResult:
    """Individual search result."""
    item_id: str
    score: float
    lexical_score: float
    semantic_score: float
    position: int
    matched_terms: list[str]
    highlights: dict[str, str]
    item: Any = None


@dataclass
class SearchResponse:
    """Search response with results and metadata."""
    query: str
    results: list[SearchResult]
    total_matches: int
    took_ms: float
    parsed_query: dict[str, Any]
    suggestions: list[str] | None = None


class BM25:
    """BM25 ranking function - industry standard for lexical search."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.doc_count, self.avgdl = 0, 0
        self.doc_lengths, self.doc_freqs, self.term_freqs, self.idf_cache = {}, {}, {}, {}
    
    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def fit(self, documents: dict[str, str]) -> "BM25":
        self.doc_count = len(documents)
        total_length = 0
        for doc_id, text in documents.items():
            tokens = self._tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            self.term_freqs[doc_id] = dict(Counter(tokens))
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        self.avgdl = total_length / self.doc_count if self.doc_count > 0 else 0
        for term, df in self.doc_freqs.items():
            self.idf_cache[term] = max(self.epsilon, math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1))
        return self
    
    def score(self, query: str, doc_id: str) -> float:
        if doc_id not in self.term_freqs:
            return 0.0
        query_terms = self._tokenize(query)
        doc_tfs = self.term_freqs[doc_id]
        doc_len = self.doc_lengths[doc_id]
        score = 0.0
        for term in query_terms:
            if term in doc_tfs:
                tf = doc_tfs[term]
                idf = self.idf_cache.get(term, self.epsilon)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
        return score
    
    def score_batch(self, query: str, doc_ids: list[str] | None = None) -> list[tuple[str, float]]:
        doc_ids = doc_ids or list(self.term_freqs.keys())
        scores = [(doc_id, self.score(query, doc_id)) for doc_id in doc_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def get_matching_terms(self, query: str, doc_id: str) -> list[str]:
        return list(set(self._tokenize(query)) & set(self.term_freqs.get(doc_id, {}).keys()))


class SemanticSearch:
    """Semantic search using dense embeddings."""
    
    def __init__(self, embedding_model: Any = None, embedding_dimension: int = 384):
        self.model = embedding_model
        self.dimension = embedding_dimension
        self.item_ids, self.embeddings, self.item_texts = [], None, {}
    
    def _get_embedding(self, text: str) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(text)
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        return np.random.randn(self.dimension).astype(np.float32)
    
    def index_items(self, items: dict[str, str]) -> "SemanticSearch":
        self.item_ids = list(items.keys())
        self.item_texts = items
        embeddings = [self._get_embedding(text) for text in items.values()]
        self.embeddings = np.array(embeddings)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)
        return self
    
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.embeddings is None:
            return []
        query_emb = self._get_embedding(query)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        similarities = self.embeddings @ query_emb
        results = list(zip(self.item_ids, similarities))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class HybridSearch:
    """Hybrid search combining BM25 + semantic."""
    
    def __init__(self, lexical_weight: float = 0.4, semantic_weight: float = 0.6, embedding_model: Any = None):
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self.bm25 = BM25()
        self.semantic = SemanticSearch(embedding_model=embedding_model)
        self.items = {}
    
    def index(self, items: list[Any], text_field: str = "description", id_field: str = "item_id") -> "HybridSearch":
        texts = {}
        for item in items:
            item_id = getattr(item, id_field, None) or item.get(id_field)
            text = getattr(item, text_field, "") or item.get(text_field, "")
            if item_id:
                texts[item_id] = text
                self.items[item_id] = item
        self.bm25.fit(texts)
        self.semantic.index_items(texts)
        return self
    
    def search(self, query: str, top_k: int = 10) -> SearchResponse:
        import time
        start = time.perf_counter()
        parsed = {"original": query, "tokens": query.lower().split()}
        
        lex_results = dict(self.bm25.score_batch(query))
        sem_results = dict(self.semantic.search(query, len(self.items)))
        
        def normalize(scores):
            if not scores:
                return {}
            vals = list(scores.values())
            mn, mx = min(vals), max(vals)
            return {k: (v - mn) / (mx - mn) if mx != mn else 0.5 for k, v in scores.items()}
        
        lex_norm = normalize(lex_results)
        sem_norm = normalize(sem_results)
        
        combined = {}
        for iid in set(lex_norm.keys()) | set(sem_norm.keys()):
            combined[iid] = self.lexical_weight * lex_norm.get(iid, 0) + self.semantic_weight * sem_norm.get(iid, 0)
        
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for pos, (iid, score) in enumerate(sorted_results, 1):
            results.append(SearchResult(
                item_id=iid, score=score, lexical_score=lex_norm.get(iid, 0), semantic_score=sem_norm.get(iid, 0),
                position=pos, matched_terms=self.bm25.get_matching_terms(query, iid),
                highlights={"description": "..."}, item=self.items.get(iid)))
        
        return SearchResponse(query=query, results=results, total_matches=len(combined),
                             took_ms=(time.perf_counter() - start) * 1000, parsed_query=parsed, suggestions=[])


class QueryUnderstanding:
    """Query understanding and intent detection."""
    
    INTENT_PATTERNS = {
        "price_intent": ["cheap", "budget", "affordable", "expensive", "luxury"],
        "location_intent": ["near", "downtown", "beachfront", "airport"],
        "amenity_intent": ["pool", "spa", "gym", "wifi", "parking", "pet"],
    }
    
    def analyze(self, query: str) -> dict[str, Any]:
        tokens = query.lower().split()
        intents = {intent: True for intent, patterns in self.INTENT_PATTERNS.items() if any(p in query.lower() for p in patterns)}
        price_range = (0, 100) if any(w in query.lower() for w in ["budget", "cheap"]) else ((200, 1000) if "luxury" in query.lower() else None)
        return {"original_query": query, "cleaned_query": query.lower(), "intents": intents,
                "entities": {"price_range": price_range}, "suggested_filters": []}
