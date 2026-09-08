from __future__ import annotations

from typing import Any

from ollama import Client, ResponseError

DEFAULT_MODEL = "gemma3:1b"

class OllamaProvider:
    def __init__(
            self,
            model: str = DEFAULT_MODEL,
            host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.client = Client(host=host)

    def chat(
            self,
            messages: list[dict[str, str]],
            temperature: float = 0.3,
    ) -> str:
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                },
            )

            return response.message.content or ""

        except ResponseError as exc:
            raise RuntimeError(
                f"Ollama request failed: {exc.error}"
            ) from exc

    def generate(
            self,
            prompt: str,
            system: str | None = None,
            temperature: float = 0.3,
    ) -> str:
        messages: list[dict[str, str]] = []

        if system:
            messages.append(
                {
                    "role": "system",
                    "content": system,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return self.chat(
            messages=messages,
            temperature=temperature,
        )

if __name__ == "__main__":
    provider = OllamaProvider()

    answer = provider.generate(
        prompt="Explain cosine similarity.",
        system="You are an AI engineering tutor.",
    )

    print(answer)
    