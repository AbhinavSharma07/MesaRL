"""Builds the chat LLM from config.json's llm_backend (same pattern as
Pattern-Recognition's agents/llm_factory.py): provider="groq" points
ChatOpenAI at Groq's OpenAI-compatible endpoint (GROQ_API_KEY), "ollama"
uses a local or cloud Ollama server, anything else falls back to OpenAI."""

import os
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

try:
    from langchain_ollama import ChatOllama
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        ChatOllama = None

OLLAMA_CLOUD_BASE_URL = "https://ollama.com"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def build_chat_llm(config: Optional[Dict[str, Any]] = None):
    config = config or {}
    llm_config = config.get("llm_backend", {"provider": "groq", "model": "openai/gpt-oss-120b"})

    if llm_config.get("provider") == "groq":
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("Groq support requires a GROQ_API_KEY environment variable.")
        return ChatOpenAI(
            model=llm_config.get("model", "openai/gpt-oss-120b"),
            temperature=0.2,
            # Explicit cap: without one, the reserved completion budget counts
            # against Groq's free-tier TPM limit (8000) alongside the prompt,
            # and the audit pipeline's later stages carry cumulative context.
            max_tokens=llm_config.get("max_tokens", 1500),
            api_key=groq_api_key,
            base_url=llm_config.get("base_url", GROQ_BASE_URL),
        )

    if llm_config.get("provider") == "ollama":
        if ChatOllama is None:
            raise RuntimeError(
                "Ollama support requires 'langchain-ollama'. Install it with: pip install langchain-ollama"
            )
        ollama_api_key = os.getenv("OLLAMA_API_KEY")
        base_url = llm_config.get("base_url")
        client_kwargs = {}
        if ollama_api_key:
            base_url = base_url or OLLAMA_CLOUD_BASE_URL
            client_kwargs["headers"] = {"Authorization": f"Bearer {ollama_api_key}"}
        return ChatOllama(
            model=llm_config.get("model", "llama3"), temperature=0.2, base_url=base_url, client_kwargs=client_kwargs
        )

    return ChatOpenAI(model=llm_config.get("model", "gpt-4o-mini"), temperature=0.2)
