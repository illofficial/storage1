from datetime import date
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from app.config import settings
from app.services.agent_core import MAX_TRANSACTIONS_PER_RESPONSE

# Type aliases for better readability
TransactionCategory = Literal["transfer", "payment", "withdrawal", "deposit", "fee"]
FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


class FintechTransactionQuery(BaseModel):
    """
    Validated arguments for the fintech database lookup tool.
    
    Examples:
        >>> query = FintechTransactionQuery(
        ...     account_id="ACC123",
        ...     start_date=date(2026, 7, 1),
        ...     end_date=date(2026, 7, 31),
        ...     category="payment",
        ...     limit=20,
        ... )
    """

    account_id: str = Field(
        ...,
        min_length=1,
        pattern=r"^ACC\d+$",
        description="Account identifier (format: ACC followed by digits)",
        examples=["ACC123", "ACC456"],
    )
    start_date: date = Field(
        ...,
        description="Start date for transaction search",
    )
    end_date: date = Field(
        ...,
        description="End date for transaction search (inclusive)",
    )
    category: Literal["all", "transfer", "payment", "withdrawal"] = Field(
        default="all",
        description="Filter transactions by category",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=MAX_TRANSACTIONS_PER_RESPONSE,
        description=f"Maximum number of transactions to return (max: {MAX_TRANSACTIONS_PER_RESPONSE})",
    )

    @model_validator(mode="after")
    def validate_date_range(self) -> Self:
        """Ensure end_date is not before start_date."""
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        return (
            f"<FintechTransactionQuery account={self.account_id} "
            f"from={self.start_date} to={self.end_date} "
            f"category={self.category}>"
        )


class TransactionRecord(BaseModel):
    """
    A single transaction row returned by the fintech lookup tool.
    
    Examples:
        >>> tx = TransactionRecord(
        ...     id="TXN-1001",
        ...     account_id="ACC123",
        ...     date=date(2026, 7, 1),
        ...     amount=-45.20,
        ...     currency="USD",
        ...     category="payment",
        ...     merchant="Cloud Services Inc.",
        ... )
    """

    id: str = Field(
        ...,
        description="Unique transaction identifier",
        examples=["TXN-1001"],
    )
    account_id: str = Field(
        ...,
        description="Account identifier",
        examples=["ACC123"],
    )
    date: date = Field(
        ...,
        description="Transaction date",
    )
    amount: float = Field(
        ...,
        description="Transaction amount (positive = credit, negative = debit)",
    )
    currency: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="ISO currency code",
        examples=["USD", "EUR"],
    )
    category: TransactionCategory = Field(
        ...,
        description="Transaction category",
        examples=["payment", "transfer"],
    )
    merchant: str = Field(
        ...,
        description="Merchant or counterparty name",
        examples=["Amazon", "Payroll"],
    )

    def is_credit(self) -> bool:
        """Check if transaction is a credit (positive amount)."""
        return self.amount > 0

    def is_debit(self) -> bool:
        """Check if transaction is a debit (negative amount)."""
        return self.amount < 0

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        return f"<Transaction {self.id} {self.merchant} {self.amount}{self.currency}>"


class UserRequest(BaseModel):
    """
    Inbound payload for the public `/v1/chat` endpoint.
    
    Examples:
        >>> req = UserRequest(message="Show me my transactions for July 2026")
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=8_000,
        description="User's natural language query",
        examples=["Show me my transactions for July 2026"],
    )


class AgentRequest(BaseModel):
    """
    Internal request model for the agent orchestrator.
    
    Allows overriding system prompt and model per request.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="User query for the agent",
        examples=["What was my total spending in July?"],
    )
    model: str = Field(
        default=settings.OPENAI_MODEL,
        min_length=1,
        description="OpenAI model to use for this request",
    )
    system_prompt: str = Field(
        default=(
            "You are a fintech assistant. Use available tools to fetch "
            "transaction data when needed, then summarize findings clearly."
        ),
        min_length=1,
        description="System prompt for the agent",
    )


class AgentResponse(BaseModel):
    """
    Response from the agent orchestrator after completing the tool loop.
    
    Contains the final answer and execution metadata.
    """

    content: str = Field(
        ...,
        description="Final answer from the agent",
    )
    model: str = Field(
        ...,
        description="Model that generated the response",
    )
    iterations: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of tool-calling iterations used",
    )
    finish_reason: FinishReason | None = Field(
        default=None,
        description="Reason the model stopped generating tokens",
    )


class ErrorResponse(BaseModel):
    """
    Standard error response format for API errors.
    """

    error: str = Field(
        ...,
        description="Human-readable error message",
    )
    error_type: str = Field(
        ...,
        description="Error type for programmatic handling",
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error details",
    )
    request_id: str | None = Field(
        default=None,
        description="Request ID for tracking",
    )


class MaxIterationsExceededError(Exception):
    """
    Raised when the agent tool loop exceeds the allowed iteration limit.
    
    This is a hard cap to prevent infinite loops and excessive token usage.
    """

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        super().__init__(f"Agent exceeded maximum iterations ({max_iterations})")

    def __str__(self) -> str:
        return f"MaxIterationsExceededError: exceeded limit of {self.max_iterations} iterations"

    def __repr__(self) -> str:
        return f"MaxIterationsExceededError(max_iterations={self.max_iterations})"


class RAGContext(BaseModel):
    """
    Retrieved context for RAG operations.
    
    Used for debugging and monitoring retrieval quality.
    """

    documents: list[str] = Field(
        ...,
        description="Retrieved document contents",
    )
    scores: list[float] | None = Field(
        default=None,
        description="Relevance scores for each document",
    )
    sources: list[str] | None = Field(
        default=None,
        description="Source identifiers for each document",
    )

    @property
    def top_document(self) -> str | None:
        """Return the highest-scoring document."""
        return self.documents[0] if self.documents else None

    def __len__(self) -> int:
        return len(self.documents)


class StreamChunk(BaseModel):
    """
    Streaming response chunk for real-time updates.
    """

    content: str = Field(
        ...,
        description="Text content chunk",
    )
    is_final: bool = Field(
        default=False,
        description="Whether this is the final chunk",
    )
    finish_reason: FinishReason | None = Field(
        default=None,
        description="Final finish reason (only set when is_final=True)",
    )
    token_count: int | None = Field(
        default=None,
        description="Cumulative token count (approximate)",
    )
