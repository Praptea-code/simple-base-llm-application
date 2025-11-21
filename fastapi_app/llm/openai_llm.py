"""OpenAI LLM implementation"""

from typing import Optional, AsyncIterator
from openai import AsyncOpenAI
from ..config import settings
from ..logging_config import get_logger
from .base import BaseLLM, LLMResponse, LLMState

logger = get_logger(__name__)


class OpenAI_LLM(BaseLLM):
    """OpenAI LLM provider implementation"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize OpenAI LLM"""
        # Get API key from settings if not provided
        api_key = api_key or getattr(settings, "openai_api_key", None)
        super().__init__(model=model, api_key=api_key, **kwargs)
        
        if not api_key:
            logger.warning("OPENAI_API_KEY not configured")
            self._set_state(LLMState.NOT_CONFIGURED, "OPENAI_API_KEY not configured")
        else:
            try:
                # Initialize async OpenAI client
                self.client = AsyncOpenAI(api_key=self.api_key)
                logger.info(f"OpenAI LLM client initialized with model: {model}")
                self._set_state(LLMState.UNHEALTHY, "Not yet validated")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self._set_state(LLMState.UNHEALTHY, str(e))

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response using OpenAI"""
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            **kwargs,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        # Use async API
        response = await self.client.chat.completions.create(**params)

        # Extract usage
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        # Extract finish reason
        finish_reason = None
        if response.choices and len(response.choices) > 0:
            finish_reason = response.choices[0].finish_reason

        # Extract content
        content = ""
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content or ""

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            metadata={"finish_reason": finish_reason},
        )

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using OpenAI"""
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        stream = await self.client.chat.completions.create(**params)

        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content