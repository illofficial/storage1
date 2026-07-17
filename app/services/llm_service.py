import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from openai import (
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    BadRequestError,
    NOT_GIVEN,
)
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
    """
    Resilient, decoupled wrapper around the OpenAI chat completions API.
    
    Provides both streaming and non-streaming responses with full support
    for tool-calling state and proper resource management.
    """

    def __init__(self, client: AsyncOpenAI) -> None:
        """
        Initialize the LLM service.
        
        Args:
            client: AsyncOpenAI client instance (shared across requests)
        """
        self._client = client

    @staticmethod
    def _handle_bad_request(exc: BadRequestError, context: str) -> None:
        """
        Unified handling of BadRequestError from OpenAI.
        
        Logs specific error types with appropriate context.
        Always re-raises the exception.
        """
        error_str = str(exc)
        
        if "context_length_exceeded" in error_str:
            logger.warning(
                "Context window exceeded during %s. Error: %s",
                context,
                error_str,
            )
        elif "invalid_parameter" in error_str:
            logger.warning(
                "Invalid parameter during %s: %s",
                context,
                error_str,
            )
        elif "invalid_request_error" in error_str:
            logger.warning(
                "Invalid request during %s: %s",
                context,
                error_str,
            )
        else:
            logger.warning(
                "BadRequestError during %s: %s",
                context,
                error_str,
            )
        
        # Always re-raise to let caller handle the error appropriately
        raise

    @staticmethod
    def _to_message_params(request: LLMRequest) -> list[ChatCompletionMessageParam]:
        """
        Convert internal models to OpenAI payloads without shedding tool metadata.
        
        Preserves:
            - `name` for function/tool messages
            - `tool_call_id` for tool response messages
        """
        result = []
        for message in request.messages:
            msg: dict[str, Any] = {
                "role": message.role,
                "content": message.content,
            }
            
            # Preserve tool metadata for function calling
            if hasattr(message, "name") and message.name:
                msg["name"] = message.name
                
            if message.role == "tool" and hasattr(message, "tool_call_id") and message.tool_call_id:
                msg["tool_call_id"] = message.tool_call_id
                
            result.append(cast(ChatCompletionMessageParam, msg))
        
        return result

    @retry_on_rate_limit
    async def _create_completion(self, request: LLMRequest) -> ChatCompletion:
        """
        Create a non-streaming chat completion.
        
        This method is wrapped with retry logic for transient failures.
        """
        max_tokens: int | NOT_GIVEN = NOT_GIVEN if request.max_tokens is None else request.max_tokens
        
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
        max_tokens: int | NOT_GIVEN,
    ) -> AsyncStream[ChatCompletionChunk]:
        """
        Open a streaming connection to OpenAI.
        
        This method is wrapped with retry logic for transient failures.
        """
        return await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

    @staticmethod
    async def _iter_deltas(stream: AsyncStream[ChatCompletionChunk]) -> AsyncIterator[str]:
        """
        Iterate over content deltas from a streaming response.
        
        Yields only text content deltas, skipping empty chunks and non-content events.
        """
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
        """
        Unified streaming logic with proper resource management.
        
        Uses async context manager to ensure HTTP connections are closed
        even when clients disconnect early.
        
        Args:
            model: OpenAI model to use
            messages: List of message parameters
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate (None = no limit)
        
        Yields:
            Token deltas as they arrive from the API
        """
        max_tokens_param: int | NOT_GIVEN = NOT_GIVEN if max_tokens is None else max_tokens
        
        logger.debug(
            "Starting stream: model=%s, temp=%.2f, max_tokens=%s",
            model,
            temperature,
            max_tokens or "unlimited",
        )
        
        try:
            # Context manager ensures HTTP connection closes cleanly
            # even under early client aborts
            async with await self._open_stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens_param,
            ) as stream:
                async for delta in self._iter_deltas(stream):
                    yield delta
                    
        except APITimeoutError as exc:
            logger.warning(
                "OpenAI API timeout during streaming for model=%s: %s",
                model,
                exc,
            )
            raise
            
        except BadRequestError as exc:
            self._handle_bad_request(exc, "streaming")
            raise  # Never reached, but keeps linter happy
            
        except APIError as exc:
            logger.exception(
                "OpenAI infrastructure error during streaming for model=%s",
                model,
            )
            raise
            
        logger.debug("Stream completed successfully for model=%s", model)

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a blocking chat completion response.
        
        Args:
            request: LLMRequest containing messages and parameters
        
        Returns:
            LLMResponse with generated content and metadata
        
        Raises:
            BadRequestError: If request parameters are invalid
            APIError: If OpenAI API returns an error
        """
        logger.debug(
            "Generating response: model=%s, max_tokens=%s",
            request.model,
            request.max_tokens or "unlimited",
        )
        
        try:
            response = await self._create_completion(request)
            
        except BadRequestError as exc:
            self._handle_bad_request(exc, "generation")
            raise  # Never reached, but keeps linter happy
            
        except APIError:
            logger.exception(
                "OpenAI infrastructure error during generation for model=%s",
                request.model,
            )
            raise

        choice = response.choices[0]
        
        # Log token usage if available
        usage_info = ""
        if hasattr(response, 'usage') and response.usage:
            usage_info = f", tokens_used={response.usage.total_tokens}"
            logger.debug(
                "Generation completed: finish_reason=%s%s",
                choice.finish_reason,
                usage_info,
            )
        
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
        )

    async def stream_response(self, request: LLMRequest) -> AsyncIterator[str]:
        """
        Stream token deltas for a standard LLMRequest.
        
        Args:
            request: LLMRequest containing messages and parameters
        
        Yields:
            Token deltas as they arrive from the API
        """
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
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream token deltas for a pre-built list of chat message params.
        
        Used by the agent flow to stream final answers after tool resolution.
        
        Args:
            messages: Pre-built list of message parameters (including tool results)
            model: OpenAI model to use (default: gpt-4o-mini)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (None = no limit)
        
        Yields:
            Token deltas as they arrive from the API
        """
        async for delta in self._stream_with_params(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield delta
