import time
from ollama_chat_client import OllamaProvider

DEFAULT_MODEL = "gemma3:1b"


provider = OllamaProvider(
    model="gemma3:1b",
)

def call_llm(prompt: dict) -> dict:
    messages = [
        {
            "role": "system",
            "content": prompt["system"],
        },
        {
            "role": "user",
            "content": prompt["user"],
        },
    ]

    start = time.perf_counter()

    response = provider.client.chat(
        model=provider.model,
        messages=messages,
        options={
            "temperature": prompt["temperature"],
        },
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return {
        "response": response.message.content or "",
        "latency_ms": round(elapsed_ms, 2),
        "model": provider.model,
    }


