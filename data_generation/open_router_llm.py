import os
import logging
from openai import OpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential_jitter,
    RetryCallState,
)
import openai


def log_retry_error(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    logging.error(
        f"OpenAI Error: {exception}. Retrying attempt {retry_state.attempt_number}..."
    )


retryable_exceptions = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
)


class OpenRouterOpenAI(DeepEvalBaseLLM):
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize by reading OPENAI_API_KEY and OPENAI_API_BASE from environment.

        - model_name: e.g. "gpt-4o" or "gpt-3.5-turbo"
        - kwargs: any additional OpenAI parameters (temperature, max_tokens, etc.)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")
        if api_key is None or api_base is None:
            raise ValueError(
                "Both OPENAI_API_KEY and OPENAI_API_BASE must be set in the environment"
            )

        self.client = OpenAI(api_key=api_key, base_url=api_base.rstrip("/"))
        self.model_name = model_name
        self.request_kwargs = kwargs

    def load_model(self):
        """
        Return a dict of parameters for chat completion.
        """
        return {"model": self.model_name, **self.request_kwargs}

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def generate(self, prompt: str) -> str:
        """
        Synchronous generation with automatic retry on rate limits.
        """
        params = self.load_model()
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], **params
        )
        return response.choices[0].message.content.strip()

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronous generation with automatic retry on rate limits.
        """
        params = self.load_model()
        response = await self.client.chat.completions.acreate(
            messages=[{"role": "user", "content": prompt}], **params
        )
        return response.choices[0].message.content.strip()

    def get_model_name(self):
        return f"OpenRouter/OpenAI ({self.model_name})"

