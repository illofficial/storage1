from fastapi import Request

from app.config import Settings
from app.services.agent_core import AgentOrchestrator
from app.services.llm_service import LLMService
from app.services.vector_db import RAGService


def get_settings(request: Request) -> Settings:
    """
    Return the application settings created during startup.
    
    Settings are loaded once from environment and cached.
    """
    return request.app.state.settings


def get_llm_service(request: Request) -> LLMService:
    """
    Return the process-wide LLMService built during startup.
    
    The service wraps an AsyncOpenAI client and handles all LLM interactions.
    """
    if not hasattr(request.app.state, "llm_service"):
        raise RuntimeError("LLMService not initialized. Check lifespan setup.")
    return request.app.state.llm_service


def get_agent_orchestrator(request: Request) -> AgentOrchestrator:
    """
    Return the process-wide AgentOrchestrator built during startup.
    
    The orchestrator runs the tool-calling loop and resolves user requests.
    """
    if not hasattr(request.app.state, "agent_orchestrator"):
        raise RuntimeError("AgentOrchestrator not initialized. Check lifespan setup.")
    return request.app.state.agent_orchestrator


def get_rag_service(request: Request) -> RAGService:
    """
    Return the process-wide RAGService built during startup.
    
    The service handles hybrid vector/keyword search with automatic fallback
    to in-memory store when Qdrant is unavailable.
    """
    if not hasattr(request.app.state, "rag_service"):
        raise RuntimeError("RAGService not initialized. Check lifespan setup.")
    return request.app.state.rag_service


# Optional: Alias for convenience in routers
# from app.dependencies import get_agent_orchestrator
