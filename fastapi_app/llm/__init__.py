"""LLM abstraction layer for different providers"""

from .base import BaseLLM, LLMResponse, LLMState
from .openai_llm import OpenAI_LLM
from .gemini import GeminiLLM
from .groq import GroqLLM

# Instantiate all providers once at import time
openai = OpenAI_LLM()
gemini = GeminiLLM()
groq = GroqLLM()

# Global registry used by the entire app
PROVIDERS: dict[str, BaseLLM] = {"openai": openai,"gemini": gemini,"groq": groq,}

__all__ = ["BaseLLM","LLMResponse","LLMState","OpenAI_LLM","GeminiLLM","GroqLLM","PROVIDERS",]