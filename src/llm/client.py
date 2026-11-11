import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "model_env": "OLLAMA_MODEL",
    },
}


def get_provider(provider=None):
    return provider or os.getenv("LLM_PROVIDER", "openrouter")


def get_model(provider=None):
    return os.getenv(PROVIDERS[get_provider(provider)]["model_env"])


def chat(messages, provider=None, model=None, temperature=0):
    settings = PROVIDERS[get_provider(provider)]
    api_key = os.getenv(settings["key_env"]) if settings["key_env"] else "ollama"
    client = OpenAI(base_url=settings["base_url"], api_key=api_key)
    response = client.chat.completions.create(
        model=model or get_model(provider),
        messages=messages,
        temperature=temperature,
    )
    choice = response.choices[0]
    if not choice.message.content:
        raise RuntimeError(f"empty answer, finish_reason={choice.finish_reason}")
    return choice.message.content