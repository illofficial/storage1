import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from openai import APIError

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
    payload: UserRequest,
    agent: AgentDep,
    llm_service: LLMServiceDep,
) -> StreamingResponse:
    """Resolve the request through the agent's tool loop and stream the answer back.

    The :class:`AgentOrchestrator` performs any tool calls and returns a message
    context; :class:`LLMService` then streams the final natural-language answer to
    the client token-by-token via ``StreamingResponse``.
    """
    try:
        context = await agent.build_context(payload.message)
    except MaxIterationsExceededError as exc:
        logger.warning("Agent exceeded its iteration limit while handling a chat request")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The agent could not complete the request within its step budget.",
        ) from exc
    except APIError as exc:
        logger.exception("Upstream LLM error while building the agent context")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream language model error.",
        ) from exc

    return StreamingResponse(
        llm_service.stream_completion(context, model=agent.model),
        media_type="text/event-stream",
    )
