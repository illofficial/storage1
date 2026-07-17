import asyncio
import logging
import math
import os
import re

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import APIError, AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    PayloadSchemaType,
    TextIndexParams,
    TokenizerType,
    VectorParams,
)

from app.models.rag import RAGConfig, RetrieveContextRequest, ScoredDocument
from app.services.retry import retry_on_rate_limit

from typing import Callable

logger = logging.getLogger(__name__)

# Errors that indicate Qdrant is unreachable or misconfigured
QDRANT_CONNECTION_ERRORS = (
    ResponseHandlingException,
    UnexpectedResponse,
    httpx.HTTPError,
    OSError,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_MOCK_SEED_DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "doc-001",
        "content": (
            "Wire transfer limits for retail accounts: daily outbound limit is "
            "USD 25,000 and monthly limit is USD 100,000."
        ),
    },
    {
        "id": "doc-002",
        "content": (
            "KYC refresh policy requires identity verification every 24 months "
            "for high-risk customer segments."
        ),
    },
    {
        "id": "doc-003",
        "content": (
            "Chargeback handling SLA: merchant disputes must be acknowledged "
            "within 2 business days and resolved within 15 business days."
        ),
    },
    {
        "id": "doc-004",
        "content": (
            "AML monitoring rules flag transactions above USD 10,000 and "
            "unusual cross-border payment patterns."
        ),
    },
    {
        "id": "doc-005",
        "content": (
            "API rate limits for fintech partners: 600 requests per minute "
            "with burst capacity up to 1200 requests per minute."
        ),
    },
]


def _tokenize(text: str) -> set[str]:
    """Tokenize text for keyword search."""
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _keyword_score(query: str, content: str) -> float:
    """BM25-inspired keyword relevance score."""
    query_tokens = _tokenize(query)
    content_tokens = _tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0
    overlap = query_tokens & content_tokens
    return len(overlap) / math.sqrt(len(content_tokens)) if overlap else 0.0


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


@dataclass(slots=True)
class StoredDocument:
    """In-memory document with cached embedding."""

    id: str
    content: str
    embedding: list[float] = field(default_factory=list)


class VectorStoreBackend(ABC):
    """Abstract interface for vector store backends."""

    @abstractmethod
    async def vector_search(self, vector: list[float], limit: int) -> list[ScoredDocument]:
        """Semantic search using vector similarity."""
        raise NotImplementedError

    @abstractmethod
    async def keyword_search(self, query: str, limit: int) -> list[ScoredDocument]:
        """Keyword-based search using full-text index."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""
        raise NotImplementedError


class EmbeddingProvider:
    """OpenAI embedding provider with retry support."""

    def __init__(self, client: AsyncOpenAI, *, model: str) -> None:
        self._client = client
        self._model = model
        self._cache: dict[int, list[float]] = {}  # Simple in-memory cache

    @retry_on_rate_limit
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text with caching."""
        cache_key = hash(text)
        if cache_key in self._cache:
            logger.debug("Embedding cache hit for text length %d", len(text))
            return self._cache[cache_key]

        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        embedding = response.data[0].embedding
        self._cache[cache_key] = embedding
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in parallel."""
        tasks = [self.embed(text) for text in texts]
        return await asyncio.gather(*tasks)


class MockVectorStore(VectorStoreBackend):
    """
    In-memory vector store with transactional initialization.
    
    Used as fallback when Qdrant is unavailable or in tests.
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        seed_documents: list[dict[str, str]] | None = None,
    ) -> None:
        self._embedder = embedder
        self._documents: list[StoredDocument] = []
        self._seed_documents = seed_documents or _MOCK_SEED_DOCUMENTS
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        """Lazy initialization with transaction-like rollback on failure."""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            try:
                logger.info(
                    "Generating embeddings for %d seed documents...",
                    len(self._seed_documents),
                )
                contents = [doc["content"] for doc in self._seed_documents]
                embeddings = await self._embedder.embed_batch(contents)

                # Atomic replacement to prevent partial state
                new_documents = []
                for doc, embedding in zip(self._seed_documents, embeddings, strict=True):
                    new_documents.append(
                        StoredDocument(
                            id=doc["id"],
                            content=doc["content"],
                            embedding=embedding,
                        )
                    )

                self._documents = new_documents
                self._initialized = True
                logger.info(
                    "MockVectorStore successfully initialized with %d documents",
                    len(self._documents),
                )
            except Exception:
                logger.exception("Failed to initialize MockVectorStore")
                self._documents = []  # Clean rollback
                self._initialized = False
                raise

    async def _search(
        self,
        scorer: Callable,
        source: str,
        limit: int,
    ) -> list[ScoredDocument]:
        """Generic search with single-pass scoring."""
        await self._ensure_initialized()

        scored_items = []
        for item in self._documents:
            score = scorer(item)
            if score > 0.0:
                scored_items.append((item, score))

        scored_items.sort(key=lambda x: x[1], reverse=True)

        return [
            ScoredDocument(
                id=item.id,
                content=item.content,
                score=score,
                source=source,
            )
            for item, score in scored_items[:limit]
        ]

    async def vector_search(self, vector: list[float], limit: int) -> list[ScoredDocument]:
        """Vector similarity search."""
        return await self._search(
            scorer=lambda item: _cosine_similarity(vector, item.embedding),
            source="vector",
            limit=limit,
        )

    async def keyword_search(self, query: str, limit: int) -> list[ScoredDocument]:
        """Keyword search."""
        return await self._search(
            scorer=lambda item: _keyword_score(query, item.content),
            source="keyword",
            limit=limit,
        )

    async def refresh(self, documents: list[dict[str, str]] | None = None) -> None:
        """Refresh in-memory documents."""
        if documents is not None:
            self._seed_documents = documents
        self._initialized = False
        self._documents = []
        await self._ensure_initialized()

    async def close(self) -> None:
        """No-op for in-memory store."""
        return None


class QdrantVectorStore(VectorStoreBackend):
    """
    Production Qdrant driver with auto-provisioning and full-text search support.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        collection_name: str,
        content_field: str = "content",
        vector_size: int = 1536,  # Default for text-embedding-3-small
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._content_field = content_field
        self._vector_size = vector_size
        self._ready = False
        self._ready_lock = asyncio.Lock()

    async def _ensure_ready(self) -> None:
        """Ensure collection and indexes exist."""
        if self._ready:
            return

        async with self._ready_lock:
            if self._ready:
                return

            try:
                # Check if collection exists
                collections = await self._client.get_collections()
                collection_names = [c.name for c in collections.collections]

                if self._collection_name not in collection_names:
                    logger.info(
                        "Creating collection '%s' with vector size %d",
                        self._collection_name,
                        self._vector_size,
                    )
                    await self._client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=VectorParams(
                            size=self._vector_size,
                            distance=Distance.COSINE,
                        ),
                    )

                # Check if text index exists
                indexes = await self._client.list_payload_indexes(self._collection_name)
                has_text_index = any(
                    idx.field_name == self._content_field
                    and idx.field_type == PayloadSchemaType.TEXT
                    for idx in indexes
                )

                if not has_text_index:
                    logger.info(
                        "Creating text index on field '%s'",
                        self._content_field,
                    )
                    await self._client.create_payload_index(
                        collection_name=self._collection_name,
                        field_name=self._content_field,
                        field_type=PayloadSchemaType.TEXT,
                        params=TextIndexParams(
                            tokenizer=TokenizerType.WORD,
                            lowercase=True,
                            min_token_len=2,
                            max_token_len=20,
                        ),
                    )

                self._ready = True
                logger.info("Qdrant store ready for collection '%s'", self._collection_name)

            except Exception:
                logger.exception("Failed to initialize Qdrant store")
                raise

    async def vector_search(self, vector: list[float], limit: int) -> list[ScoredDocument]:
        """Semantic search using vector similarity."""
        await self._ensure_ready()

        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )

        points = response.points if hasattr(response, "points") else response
        results = []
        for point in points:
            payload = point.payload or {}
            content = payload.get(self._content_field)
            if isinstance(content, str):
                results.append(
                    ScoredDocument(
                        id=str(point.id),
                        content=content,
                        score=float(point.score or 0.0),
                        source="vector",
                    )
                )
        return results

    async def keyword_search(self, query: str, limit: int) -> list[ScoredDocument]:
        """Full-text keyword search using Qdrant's MatchText."""
        await self._ensure_ready()

        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=None,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key=self._content_field,
                        match=MatchText(text=query),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        )

        points = response.points if hasattr(response, "points") else response
        results = []
        for point in points:
            payload = point.payload or {}
            content = payload.get(self._content_field)
            if isinstance(content, str):
                results.append(
                    ScoredDocument(
                        id=str(point.id),
                        content=content,
                        score=_keyword_score(query, content),
                        source="keyword",
                    )
                )
        return results

    async def close(self) -> None:
        """Close Qdrant client."""
        await self._client.close()


async def create_vector_store(
    config: RAGConfig,
    embedder: EmbeddingProvider,
    vector_size: int = 1536,
) -> VectorStoreBackend:
    """
    Factory for creating appropriate vector store backend.
    
    Falls back to MockVectorStore if Qdrant is unavailable or not configured.
    """
    if config.use_mock:
        logger.info("Using MockVectorStore (forced by config)")
        return MockVectorStore(embedder=embedder)

    qdrant_url = config.qdrant_url or os.getenv("QDRANT_URL")
    qdrant_api_key = config.qdrant_api_key or os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        logger.warning("QDRANT_URL not set; using MockVectorStore")
        return MockVectorStore(embedder=embedder)

    try:
        client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Test connection by listing collections
        await client.get_collections()
        
        logger.info("Connected to Qdrant at %s", qdrant_url)
        return QdrantVectorStore(
            client,
            collection_name=config.collection_name,
            vector_size=vector_size,
        )

    except QDRANT_CONNECTION_ERRORS:
        logger.exception("Failed to connect to Qdrant; falling back to MockVectorStore")
        return MockVectorStore(embedder=embedder)


class RAGService:
    """
    Hybrid search engine with reciprocal rank fusion.
    
    Combines vector and keyword search with configurable bias.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        config: RAGConfig | None = None,
        vector_store: VectorStoreBackend | None = None,
        vector_size: int = 1536,
    ) -> None:
        self._config = config or RAGConfig()
        self._client = client or AsyncOpenAI(api_key=api_key)
        self._embedder = EmbeddingProvider(self._client, model=self._config.embedding_model)
        self._vector_store = vector_store
        self._vector_size = vector_size
        self._store_lock = asyncio.Lock()

    async def _get_store(self) -> VectorStoreBackend:
        """Lazy initialization of vector store."""
        if self._vector_store is not None:
            return self._vector_store

        async with self._store_lock:
            if self._vector_store is None:
                self._vector_store = await create_vector_store(
                    self._config,
                    self._embedder,
                    self._vector_size,
                )
            return self._vector_store

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_results: list[ScoredDocument],
        keyword_results: list[ScoredDocument],
        *,
        limit: int,
        rrf_k: int,
        vector_weight: float = 0.5,
    ) -> list[ScoredDocument]:
        """
        Hybrid RRF with normalized weights.
        
        Uses pure RRF ranking with optional bias for vector results.
        """
        # Normalize weights to sum to 1.0
        total_weight = vector_weight + (1.0 - vector_weight)
        norm_vector_weight = vector_weight / total_weight if total_weight > 0 else 0.5
        norm_keyword_weight = (1.0 - vector_weight) / total_weight if total_weight > 0 else 0.5

        fused_scores: dict[str, float] = {}
        documents: dict[str, ScoredDocument] = {}

        for rank, document in enumerate(vector_results, start=1):
            score = norm_vector_weight / (rrf_k + rank)
            fused_scores[document.id] = fused_scores.get(document.id, 0.0) + score
            documents[document.id] = document

        for rank, document in enumerate(keyword_results, start=1):
            score = norm_keyword_weight / (rrf_k + rank)
            if document.id in fused_scores:
                fused_scores[document.id] += score
            else:
                fused_scores[document.id] = score
            documents.setdefault(document.id, document)

        ranked_ids = sorted(
            fused_scores.keys(),
            key=lambda doc_id: fused_scores[doc_id],
            reverse=True,
        )

        return [
            ScoredDocument(
                id=doc_id,
                content=documents[doc_id].content,
                score=fused_scores[doc_id],
                source="hybrid",
            )
            for doc_id in ranked_ids[:limit]
        ]

    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[ScoredDocument]:
        """
        Perform hybrid search combining vector and keyword methods.
        
        Falls back to pure vector search if keyword search fails.
        """
        validated = RetrieveContextRequest.model_validate({
            "query": query,
            "limit": limit,
        })

        try:
            query_vector = await self._embedder.embed(validated.query)
            candidate_limit = max(validated.limit * 2, validated.limit)
            store = await self._get_store()

            try:
                vector_results, keyword_results = await asyncio.gather(
                    store.vector_search(query_vector, candidate_limit),
                    store.keyword_search(validated.query, candidate_limit),
                )
            except QDRANT_CONNECTION_ERRORS:
                logger.warning(
                    "Qdrant connection lost; switching to MockVectorStore"
                )
                async with self._store_lock:
                    self._vector_store = MockVectorStore(embedder=self._embedder)
                store = await self._get_store()
                vector_results, keyword_results = await asyncio.gather(
                    store.vector_search(query_vector, candidate_limit),
                    store.keyword_search(validated.query, candidate_limit),
                )

            return self._reciprocal_rank_fusion(
                vector_results,
                keyword_results,
                limit=validated.limit,
                rrf_k=self._config.rrf_k,
                vector_weight=self._config.hybrid_vector_weight,
            )

        except APIError:
            logger.exception("OpenAI API error during embedding for query: %s", validated.query)
            raise

    async def retrieve_context(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[str]:
        """
        Retrieve relevant context documents for a query.
        
        Args:
            query: User query
            limit: Maximum number of documents to return.
                  If None, uses RAGConfig.default_retrieval_limit.
        """
        default_retrieval_limit: int = Field(default=3, ge=1, le=50)
        if limit is None:
            limit = self._config.default_retrieval_limit or 3

        validated = RetrieveContextRequest.model_validate({
            "query": query,
            "limit": limit,
        })
        results = await self.hybrid_search(validated.query, limit=validated.limit)
        return [result.content for result in results]

    async def health_check(self) -> dict[str, bool | str]:
        """
        Check health of vector store backend.
        
        Returns status of the underlying store.
        """
        try:
            store = await self._get_store()
            if isinstance(store, QdrantVectorStore):
                await store._ensure_ready()
                return {
                    "status": "healthy",
                    "backend": "qdrant",
                    "collection": store._collection_name,
                }
            else:
                return {
                    "status": "healthy",
                    "backend": "mock",
                    "documents": len(store._documents),
                }
        except Exception as e:
            logger.exception("Health check failed")
            return {
                "status": "unhealthy",
                "backend": "unknown",
                "error": str(e),
            }

    async def close(self) -> None:
        """Release all resources."""
        if self._vector_store is not None:
            await self._vector_store.close()
