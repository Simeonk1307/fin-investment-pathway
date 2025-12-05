from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatPerplexity
# from langchain_huggingface import HuggingFaceChat, HuggingFaceHubChat
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_groq import ChatGroq
from config.settings import LLMSettings

LLMProvider = Literal["openai", "gemini", "claude", "perplexity","huggingface", "groq"]

llm_settings = LLMSettings()

safety_model = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=2048,
                api_key=llm_settings.get_key("groq"),
            )

def get_llm(
    provider: LLMProvider = "groq",
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
    
    if provider == "openai":
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=temperature,
            api_key=llm_settings.get_key("openai"),
            **kwargs
        )
    
    elif provider == "perplexity":
        api_key = llm_settings.get_key("perplexity")
        # Use correct model name
        model_name = model or "sonar"
        
        try:
            return ChatPerplexity(
                model=model_name,
                temperature=temperature,
                pplx_api_key=api_key,
                timeout=60,  # Important!
                max_retries=2,
                **kwargs
            )
        except Exception as e:
            print(f"[ERROR] Failed to create Perplexity client: {e}")

    elif provider == "huggingface":
        hf_api_key = llm_settings.get_key("huggingface")
        if not hf_api_key:
            raise ValueError("Hugging Face API key is not set in the environment.")

        llm = HuggingFacePipeline.from_model_id(
            model_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            pipeline_kwargs=dict(
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
            ),
        )

        chat_model = ChatHuggingFace(llm=llm)
        return chat_model
    
    elif provider == "groq":
        return ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=2048,
                api_key=llm_settings.get_key("groq"),
            )
    else:
        raise ValueError(f"Unknown provider: {provider}")
    

LLM = get_llm()