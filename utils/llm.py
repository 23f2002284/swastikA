"""LLM interaction utilities.

Helper functions for working with language models.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.settings import settings

T = TypeVar("T")
R = TypeVar("R")
M = TypeVar("M", bound=BaseModel)

logger = logging.getLogger(__name__)

def estimate_token_count(text: str) -> int:
    return len(text) // 4

def get_llm(
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    completions: int = 1,
) -> BaseChatModel:
    """Get LLM with specified configuration.

    Args:
        model_name: The model to use
        temperature: Temperature for generation
        completions: How many completions we need (affects temperature for diversity)

    Returns:
        Configured LLM instance
    """
    # Use higher temp when doing multiple completions for diversity
    if completions > 1 and temperature == 0.0:
        temperature = 0.2

    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY must be set in environment variables for Vertex AI")

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=settings.gemini_api_key,
    )


def get_default_llm() -> BaseChatModel:
    """Get default LLM instance."""
    return get_llm()

async def call_llm_with_structured_output(
    llm: ChatGoogleGenerativeAI,
    output_class: Type[M],
    messages: List[Tuple[str, str]],
    context_desc: str = "",
) -> Optional[M]:
    """Call LLM with structured output and consistent error handling.

    Args:
        llm: LLM instance
        output_class: Pydantic model for structured output
        messages: Messages to send to the LLM
        context_desc: Description for error logs

    Returns:
        Structured output or None if error
    """
    try:
        return await llm.with_structured_output(output_class).ainvoke(messages)
    except Exception as e:
        logger.error(f"Error in LLM call for {context_desc}: {e}")
        return None

async def process_with_voting(
    items: List[T],
    processor: Callable[[T, Any], Awaitable[Tuple[bool, Optional[R]]]],
    llm: Any,
    completions: int,
    min_successes: int,
    result_factory: Callable[[R, T], Any],
    description: str = "item",
) -> List[Any]:
    """Process items with multiple LLM attempts and consensus voting.

    Args:
        items: Items to process
        processor: Function that processes each item
        llm: LLM instance
        completions: How many attempts per item
        min_successes: How many must succeed
        result_factory: Function to create final result
        description: Item type for logs

    Returns:
        List of successfully processed results
    """
    results = []

    for item in items:
        # Make multiple attempts concurrently
        tasks = [processor(item, llm) for _ in range(completions)]
        attempts = await asyncio.gather(*tasks)

        # Count successes
        success_count = sum(1 for success, _ in attempts if success)

        # Only proceed if we have enough successes
        if success_count < min_successes:
            logger.info(
                f"Not enough successes ({success_count}/{min_successes}) for {description}"
            )
            continue

        # Use the first successful result
        for success, result in attempts:
            if success and result is not None:
                processed_result = result_factory(result, item)
                if processed_result:
                    results.append(processed_result)
                    break

    return results