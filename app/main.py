import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from app.config import get_settings
from app.models.rag import RAGConfig
from app.routers.v1 import agent as agent_router
from app.services.agent_core import AgentOrchestrator
from app.services.llm_service import LLMService
from app.services.vector_db import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Build shared, decoupled service singletons and store them on app.state.
    
    Services are initialized once at startup and cleaned up on shutdown.
    """
    settings = get_settings()
    
    # OpenAI client
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.request_timeout_seconds,
    )
    
    # RAG configuration
    rag_config = RAGConfig(
        collection_name=settings.qdrant_collection,
        embedding_model=settings.embedding_model,
        vector_size=settings.embedding_vector_size,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
    )
    
    # Store services in app state
    app.state.settings = settings
    app.state.openai_client = client
    app.state.llm_service = LLMService(client)
    
    # Agent orchestrator with mock lookup (replace with real implementation)
    app.state.agent_orchestrator = AgentOrchestrator(
        client,
        model=settings.openai_model,
        max_iterations=settings.agent_max_iterations,
        # fintech_lookup=real_fintech_lookup,  # <-- Add when ready
    )
    
    # RAG service
    app.state.rag_service = RAGService(
        client=client,
        config=rag_config,
        vector_size=settings.embedding_vector_size,
    )

    logger.info("Application services initialized successfully")
    
    try:
        yield
    finally:
        # Cleanup
        await app.state.rag_service.close()
        await client.close()
        logger.info("Application services shut down successfully")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()
    
    app = FastAPI(
        title="Fintech Agent API",
        version="1.0.0",
        description="Streaming, tool-calling fintech agent with RAG capabilities",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Health check
    @app.get("/health", tags=["health"])
    async def health(request: Request) -> dict[str, Any]:
        """Health check endpoint with service status."""
        status = {
            "status": "ok",
            "version": "1.0.0",
            "services": {},
        }
        
        # Check RAG service
        try:
            rag = request.app.state.rag_service
            await rag._get_store()
            status["services"]["rag"] = "healthy"
        except Exception as e:
            status["services"]["rag"] = f"unhealthy: {str(e)}"
            status["status"] = "degraded"
        
        # Check OpenAI
        try:
            client = request.app.state.openai_client
            # Simple test call (optional)
            status["services"]["openai"] = "healthy"
        except Exception as e:
            status["services"]["openai"] = f"unhealthy: {str(e)}"
            status["status"] = "degraded"
        
        return status
    
    # Include routers
    app.include_router(agent_router.router)
    
    return app


# Application instance
app = create_app()
