"""Base LLM class for all LLM providers"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel
from enum import Enum


class LLMState(str, Enum):
    """LLM provider states"""
    NOT_CONFIGURED = "not_configured"  # No API key provided
    UNHEALTHY = "unhealthy"  # API key present but invalid/error
    HEALTHY = "healthy"  # API key present and working


class LLMResponse(BaseModel):
    """Response from LLM provider"""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLM(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the LLM provider
        
        Args:
            model: Model name/identifier
            api_key: API key for the provider (optional, can be set via env)
            **kwargs: Additional provider-specific configuration
        """
        self.model = model
        self.api_key = api_key
        self.config = kwargs
        self._state: LLMState = LLMState.NOT_CONFIGURED
        self._error_message: Optional[str] = None
    
    @property
    def state(self) -> LLMState:
        """Get the current state of the LLM provider"""
        return self._state
    
    @property
    def error_message(self) -> Optional[str]:
        """Get the error message if state is unhealthy"""
        return self._error_message
    
    def _set_state(self, state: LLMState, error_message: Optional[str] = None):
        """Set the state of the LLM provider"""
        self._state = state
        self._error_message = error_message
    
    def is_configured(self) -> bool:
        """
        Check if the LLM provider is configured (has API key)
        
        Returns:
            True if API key is present
        """
        return self.api_key is not None and len(self.api_key) > 0
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM
        
        Args:
            prompt: Input prompt/text
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse object with the generated content
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Generate a streaming response from the LLM
        
        Args:
            prompt: Input prompt/text
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Chunks of the generated response
        """
        pass
    
    def validate_config(self) -> LLMState:
        """
        Validate the LLM configuration
        
        Returns:
            LLMState indicating the current state
        """
        if not self.is_configured():
            self._set_state(LLMState.NOT_CONFIGURED, "API key not configured")
            return LLMState.NOT_CONFIGURED
        
        # If already configured but not healthy, preserve existing error message
        # or set a default one if none exists
        if self._state != LLMState.HEALTHY:
            if not self._error_message or self._error_message == "Configuration not validated":
                self._set_state(LLMState.UNHEALTHY, "Not yet validated")
        
        return self._state
    
    async def check_health(self) -> LLMState:
        """
        Check the health of the LLM provider by attempting a test call
        
        Returns:
            LLMState indicating the current state
        """
        if not self.is_configured():
            self._set_state(LLMState.NOT_CONFIGURED, "API key not configured")
            return LLMState.NOT_CONFIGURED
        
        try:
            # Attempt a simple test generation
            await self.generate(
                prompt="test",
                temperature=0.1,
                max_tokens=5
            )
            self._set_state(LLMState.HEALTHY)
            return LLMState.HEALTHY
        except Exception as e:
            error_msg = str(e)
            self._set_state(LLMState.UNHEALTHY, error_msg)
            return LLMState.UNHEALTHY

