from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatPerplexity
from langchain_groq import ChatGroq

import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatPerplexity
from langchain_groq import ChatGroq

LLMProvider = Literal["openai", "perplexity", "groq"]


class SafetyLLM:
    """Safeguard LLM instance using Perplexity"""
    
    def __init__(self):
        self.model = ChatPerplexity(
            model="sonar",
            temperature=0.3,
            pplx_api_key=os.getenv("PPLX_API_KEY"),
            timeout=60,
            max_retries=2
        )
    
    def get_model(self):
        return self.model


class init_LLM:
    """Main LLM class with support for multiple providers"""
    
    def __init__(
        self,
        provider: LLMProvider = "perplexity",
        model: str = None,
        temperature: float = 0.3,
        **kwargs
    ):
        self.provider = provider
        self.model_name = model
        self.temperature = temperature
        self.kwargs = kwargs
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the LLM based on provider"""
        
        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model_name or "gpt-4o-mini",
                temperature=self.temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
                **self.kwargs
            )
        
        elif self.provider == "perplexity":
            model_name = self.model_name or "sonar"
            
            try:
                return ChatPerplexity(
                    model=model_name,
                    temperature=self.temperature,
                    pplx_api_key=os.getenv("PPLX_API_KEY"),
                    timeout=60,
                    max_retries=2,
                    **self.kwargs
                )
            except Exception as e:
                print(f"[ERROR] Failed to create Perplexity client: {e}")
                raise
        
        elif self.provider == "groq":
            return ChatGroq(
                model=self.model_name or "llama-3.1-8b-instant",
                temperature=self.temperature,
                max_tokens=2048,
                api_key=os.getenv("GROQ_API_KEY"),
                **self.kwargs
            )
        
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def get_model(self):
        """Return the underlying model instance"""
        return self.model


# Legacy support - you can keep this if needed for backward compatibility
def get_llm(
    provider: LLMProvider = "perplexity",
    model: str = None,
    temperature: float = 0.3,
    **kwargs
):
    """
    Factory function to get LLM instance based on provider
    
    Usage:
        llm = get_llm("openai", model="gpt-4")
        llm = get_llm("perplexity", model="sonar")
    """
    return init_LLM(provider, model, temperature, **kwargs).get_model()


# Initialize instances for backward compatibility
safety_model = SafetyLLM().get_model()