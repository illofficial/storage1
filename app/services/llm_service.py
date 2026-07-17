import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from openai import APIError, AsyncOpenAI, AsyncStream, BadRequestError, NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)

from app.models.llm import LLMRequest, LLMResponse
from app.services.retry import retry_on_rate_limit

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7


class LLMService:
    """Resilient, decoupled wrapper around the OpenAI chat completions API with full tool-calling state support."""

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    @staticmethod
    def _to_message_params(request: LLMRequest) -> list[ChatCompletionMessageParam]:
        """Convert internal models to OpenAI payloads without shedding tool execution metadata."""
        result = []
        for message in request.messages:
            msg: dict[str, Any] = {"role": message.role, "content": message.content}
            
            # Crucial Fix: Retain tool metadata so OpenAI can link functions back to the conversation branch
            if hasattr(message, "name") and message.name:
                msg["name"] = message.name
                
            if message.role == "tool" and hasattr(message, "tool_call_id") and message.tool_call_id:
                msg["tool_call_id"] = message.tool_call_id
                
            result.append(cast(ChatCompletionMessageParam, msg))
        return result

    @retry_on_rate_limit
    async def _create_completion(self, request: LLMRequest) -> ChatCompletion:
        max_tokens: int | NotGiven = NOT_GIVEN if request.max_tokens is None else request.max_tokens
        return await self._client.chat.completions.create(
            model=request.model,
            messages=self._to_message_params(request),
            temperature=request.temperature,
            max_tokens=max_tokens,
        )

    @retry_on_rate_limit
    async def _open_stream(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        max_tokens: int | NotGiven,
    ) -> AsyncStream[ChatCompletionChunk]:
        return await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

    @staticmethod
    async def _iter_deltas(stream: AsyncStream[ChatCompletionChunk]) -> AsyncIterator[str]:
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def _stream_with_params(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Unified, DRY-compliant streaming logic protecting against socket resource leaks."""
        max_tokens_param: int | NotGiven = NOT_GIVEN if max_tokens is None else max_tokens
        
        try:
            # Context manager ensures HTTP connection closes cleanly even under early client aborts
            async with await self._open_stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens_param,
            ) as stream:
                async for delta in self._iter_deltas(stream):
                    yield delta
                    
        except BadRequestError as exc:
            if "context_length_exceeded" in str(exc):
                logger.warning("Context window bounds violated on OpenAI stream: %s", exc)
            raise
        except APIError:
            logger.exception("OpenAI infrastructure API error while streaming payload")
            raise

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate a raw, blocking chat completion response with graceful error translation."""
        try:
            response = await self._create_completion(request)
        except BadRequestError as exc:
            if "context_length_exceeded" in str(exc):
                logger.warning("Context window bounds violated on blocking generation: %s", exc)
            raise
        except APIError:
            logger.exception("OpenAI infrastructure API error while generating a response")
            raise

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
        )

    async def stream_response(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream token deltas for a standard incoming user LLMRequest."""
        async for delta in self._stream_with_params(
            model=request.model,
            messages=self._to_message_params(request),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ):
            yield delta

    async def stream_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,  # Crucial Fix: Solved the hardcoded token-spend leak
    ) -> AsyncIterator[str]:
        """Stream token deltas for a pre-built list of chat message params (agent pipelines)."""
        async for delta in self._stream_with_params(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield delta
