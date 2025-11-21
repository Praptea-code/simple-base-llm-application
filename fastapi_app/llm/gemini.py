"""Google Gemini implementation using google-genai package"""

from typing import Optional, AsyncIterator
from ..config import settings
from ..logging_config import get_logger
from .base import BaseLLM, LLMResponse, LLMState

logger = get_logger(__name__)

# Try to import google-genai (the package in uv.lock)
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    # Fallback to google-generativeai if that's what's installed
    try:
        import google.generativeai as genai_legacy
        GENAI_AVAILABLE = False
        logger.warning("Using legacy google-generativeai package")
    except ImportError:
        genai_legacy = None
        GENAI_AVAILABLE = False
        logger.error("No Google GenAI package available")


class GeminiLLM(BaseLLM):
    """Gemini Pro implementation supporting both google-genai and google-generativeai"""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
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
            if GENAI_AVAILABLE:
                # Using google-genai package
                self.client = genai.Client(api_key=api_key)
                logger.info(f"Gemini client (google-genai) initialized with model: {model}")
            elif genai_legacy:
                # Using google-generativeai package
                genai_legacy.configure(api_key=api_key)
                self.client = genai_legacy.GenerativeModel(model)
                logger.info(f"Gemini client (legacy) initialized with model: {model}")
            else:
                raise ImportError("No Google GenAI package available")
            
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

        try:
            if GENAI_AVAILABLE:
                # google-genai package API
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=kwargs.get("top_p", 0.95),
                    top_k=kwargs.get("top_k", 64),
                )
                if max_tokens:
                    config.max_output_tokens = max_tokens

                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                content = response.text

                usage = None
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    usage = {
                        "prompt_tokens": getattr(um, "prompt_token_count", 0),
                        "completion_tokens": getattr(um, "candidates_token_count", 0),
                        "total_tokens": getattr(um, "total_token_count", 0),
                    }
            else:
                # Legacy google-generativeai package
                generation_config = {
                    "temperature": temperature,
                    "top_p": kwargs.get("top_p", 0.95),
                    "top_k": kwargs.get("top_k", 64),
                }
                if max_tokens:
                    generation_config["max_output_tokens"] = max_tokens

                response = await self.client.generate_content_async(
                    prompt,
                    generation_config=generation_config,
                    stream=False,
                )
                content = response.text
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

        try:
            if GENAI_AVAILABLE:
                # google-genai package streaming
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=kwargs.get("top_p", 0.95),
                    top_k=kwargs.get("top_k", 64),
                )
                if max_tokens:
                    config.max_output_tokens = max_tokens

                async for chunk in self.client.aio.models.generate_content_stream(
                    model=self.model,
                    contents=prompt,
                    config=config,
                ):
                    if chunk.text:
                        yield chunk.text
            else:
                # Legacy streaming
                generation_config = {
                    "temperature": temperature,
                    "top_p": kwargs.get("top_p", 0.95),
                    "top_k": kwargs.get("top_k", 64),
                }
                if max_tokens:
                    generation_config["max_output_tokens"] = max_tokens

                response = await self.client.generate_content_async(
                    prompt,
                    generation_config=generation_config,
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