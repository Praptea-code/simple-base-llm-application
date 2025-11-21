# fastapi_app/llm/gemini.py
"""Real Google Gemini implementation using the official SDK (2025 version)"""

from typing import Optional, AsyncIterator, Dict, Any
import google.generativeai as genai
from ..config import settings
from ..logging_config import get_logger
from .base import BaseLLM, LLMResponse, LLMState

logger = get_logger(__name__)


class GeminiLLM(BaseLLM):
    """Proper async Gemini Pro implementation"""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        api_key = api_key or settings.gemini_api_key

        super().__init__(model=model, api_key=api_key, **kwargs)

        if not api_key:
            logger.warning("GEMINI_API_KEY not configured")
            self._set_state(LLMState.NOT_CONFIGURED, "GEMINI_API_KEY not set")
            self.client = None
            return

        try:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
            logger.info(f"Gemini client initialized with model: {model}")
            # Start as unhealthy until proven otherwise
            self._set_state(LLMState.UNHEALTHY, "Not validated yet")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self._set_state(LLMState.UNHEALTHY, str(e))
            self.client = None

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        generation_config = {
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 64),
        }
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        try:
            response = await self.client.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(**generation_config),
                stream=False,
            )

            content = response.text

            # Extract usage if available
            usage = None
            if hasattr(response, "usage_metadata"):
                um = response.usage_metadata
                usage = {
                    "prompt_tokens": um.prompt_token_count,
                    "completion_tokens": um.candidates_token_count,
                    "total_tokens": um.total_token_count,
                }

            if self.state != LLMState.HEALTHY:
                self._set_state(LLMState.HEALTHY)

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                metadata={"finish_reason": "stop"},
            )

        except Exception as e:
            error_msg = str(e)
            if "API key not valid" in error_msg or "403" in error_msg:
                self._set_state(LLMState.UNHEALTHY, "Invalid GEMINI_API_KEY")
            else:
                self._set_state(LLMState.UNHEALTHY, error_msg)
            raise

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        if not self.client:
            yield "[Error: Gemini not configured]"
            return

        generation_config = {
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 64),
        }
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        try:
            response = await self.client.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(**generation_config),
                stream=True,
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

            if self.state != LLMState.HEALTHY:
                self._set_state(LLMState.HEALTHY)

        except Exception as e:
            error_msg = str(e)
            self._set_state(LLMState.UNHEALTHY, error_msg)
            yield f"\n\n[Gemini Error: {error_msg}]"