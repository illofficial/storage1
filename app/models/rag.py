from typing import Literal, Self
from pydantic import BaseModel, Field, model_validator

class ScoredDocument(BaseModel):
    """
    A document with relevance score from a retrieval operation.
    
    Contains the document content, similarity score, and search source.
    
    Examples:
        >>> doc = ScoredDocument(
        ...     id="doc-001",
        ...     content="AML policy requires verification every 24 months.",
        ...     score=0.89,
        ...     source="hybrid",
        ... )
    """

    id: str = Field(
        ...,
        min_length=1,
        description="Document identifier",
        examples=["doc-001", "policy-aml-2024"],
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Document content (truncated to 10,000 chars)",
        examples=["AML policy requires identity verification every 24 months..."],
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score (0.0 = not relevant, 1.0 = highly relevant)",
        examples=[0.89, 0.45],
    )
    source: Literal["vector", "keyword", "hybrid"] = Field(
        default="hybrid",
        description="Search method that produced this result",
        examples=["vector", "keyword", "hybrid"],
    )

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return (
            f"<ScoredDocument id={self.id} "
            f"score={self.score:.3f} "
            f"source={self.source} "
            f"content='{content_preview}'>"
        )


class RAGConfig(BaseModel):
    """
    Configuration for the RAG (Retrieval-Augmented Generation) service.
    
    Controls vector store, embedding model, and hybrid search parameters.
    
    Examples:
        >>> config = RAGConfig(
        ...     collection_name="fintech_docs",
        ...     embedding_model="text-embedding-3-small",
        ...     hybrid_vector_weight=0.8,
        ...     rrf_k=60,
        ... )
    """

    collection_name: str = Field(
        default="documents",
        min_length=1,
        description="Qdrant collection name for storing documents",
        examples=["fintech_docs", "knowledge_base"],
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        min_length=1,
        description="OpenAI embedding model for generating vectors",
        examples=["text-embedding-3-small", "text-embedding-3-large"],
    )
    vector_size: int = Field(
        default=1536,
        gt=0,
        description="Embedding vector dimension (1536 for text-embedding-3-small)",
    )
    qdrant_url: str | None = Field(
        default=None,
        description="Qdrant server URL (None = use environment variable QDRANT_URL)",
        examples=["http://localhost:6333", "https://qdrant.cloud"],
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant API key (None = use environment variable QDRANT_API_KEY)",
    )
    use_mock: bool | None = Field(
        default=None,
        description="Force mock store when True. None auto-detects from Qdrant availability.",
    )
    hybrid_vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for vector search in hybrid scoring (0.0 = keyword only, 1.0 = vector only)",
        examples=[0.7, 0.8, 1.0],
    )
    rrf_k: int = Field(
        default=60,
        gt=0,
        description="RRF (Reciprocal Rank Fusion) constant for score aggregation",
        examples=[60, 100],
    )

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        return (
            f"<RAGConfig collection={self.collection_name} "
            f"model={self.embedding_model} "
            f"use_mock={self.use_mock}>"
        )


class RetrieveContextRequest(BaseModel):
    """
    Request to retrieve context documents for a query.
    
    Used by the RAG service to fetch relevant documents.
    
    Examples:
        >>> request = RetrieveContextRequest(
        ...     query="What are the AML requirements?",
        ...     limit=5,
        ... )
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User query for context retrieval",
        examples=["What are the AML requirements for international transfers?"],
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Maximum number of documents to return",
        examples=[3, 5, 10],
    )

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        query_preview = self.query[:50] + "..." if len(self.query) > 50 else self.query
        return f"<RetrieveContextRequest query='{query_preview}' limit={self.limit}>"
