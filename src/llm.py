from __future__ import annotations

from langchain_core.language_models import BaseChatModel

_PROVIDERS = ("openai", "openrouter", "vllm", "gemini")


def get_llm(
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    
    provider = provider.lower()

    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider={provider!r}. "
            f"Choose from: {_PROVIDERS}"
        )

    if provider in ("openai", "openrouter", "vllm"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported provider: {provider}")