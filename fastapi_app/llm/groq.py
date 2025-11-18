"""Groq LLM implementation using OpenAI-compatible API"""

from typing import Optional, AsyncIterator
from openai import AsyncOpenAI
from ..config import settings
from ..logging_config import get_logger
from .base import BaseLLM, LLMResponse, LLMState

logger = get_logger(__name__)


class GroqLLM(BaseLLM):
    """Groq LLM provider implementation using OpenAI-compatible API"""

    def __init__(
        self,
        model: str = "openai/gpt-oss-safeguard-20b",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Groq LLM

        Args:
            model: Groq model name (default: "llama-3.1-70b-versatile")
            api_key: Groq API key (defaults to GROQ_API_KEY from settings)
            **kwargs: Additional Groq configuration
        """
        api_key = api_key or getattr(settings, "groq_api_key", None)
        super().__init__(model=model, api_key=api_key, **kwargs)
        
        if not api_key:
            logger.warning("GROQ_API_KEY not configured")
            self._set_state(LLMState.NOT_CONFIGURED, "GROQ_API_KEY not configured")
        else:
            try:
                # Initialize OpenAI-compatible client with Groq's base URL
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                logger.info(f"Groq LLM client initialized with model: {model}")
                # Set to unhealthy initially, will be validated on first use or health check
                self._set_state(LLMState.UNHEALTHY, "Not yet validated")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self._set_state(LLMState.UNHEALTHY, str(e))

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response using Groq

        Args:
            prompt: Input prompt/text
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Groq parameters

        Returns:
            LLMResponse object with the generated content
        """
        # Prepare generation parameters
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            **kwargs,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        # Generate response using async API
        response = await self.client.chat.completions.create(**params)

        # Extract usage information
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
            metadata={
                "finish_reason": finish_reason,
            },
        )

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response using Groq

        Args:
            prompt: Input prompt/text
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Groq parameters

        Yields:
            Chunks of the generated response
        """
        # Prepare generation parameters
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        # Generate streaming response
        stream = await self.client.chat.completions.create(**params)

        # Yield chunks asynchronously
        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

    def validate_config(self) -> LLMState:
        """Validate Groq configuration"""
        return super().validate_config()


