import asyncio
import logging
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from openai import APIError, BadRequestError

from app.dependencies import get_agent_orchestrator, get_llm_service
from app.models.agent import MaxIterationsExceededError, UserRequest
from app.services.agent_core import AgentOrchestrator
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["agent"])

AgentDep = Annotated[AgentOrchestrator, Depends(get_agent_orchestrator)]
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]


@router.post("/chat")
async def chat(
    request: Request,
    payload: UserRequest,
    agent: AgentDep,
    llm_service: LLMServiceDep,
) -> StreamingResponse:
    """
    Resolve the request through the agent's tool loop and stream the answer back.

    The agent performs any necessary tool calls and returns a message context,
    then the LLM service streams the final natural-language answer to the client.

    ## Flow:
    1. Agent builds context with tool resolution
    2. LLM service streams the final answer
    3. Client receives tokens via Server-Sent Events

    ## Error Responses:
    - 400: Invalid request or context too long
    - 429: Agent exceeded iteration limit
    - 502: OpenAI API error
    - 504: Agent execution timeout
    - 499: Client disconnected

    ## Example:
        ```bash
        curl -N -X POST http://localhost:8000/v1/chat \\
          -H 'Content-Type: application/json' \\
          -d '{"message": "Show me my transactions for July 2026"}'
