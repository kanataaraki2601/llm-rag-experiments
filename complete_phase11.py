from __future__ import annotations

import time
from typing import Any

from ollama import Client


MODEL = "gemma3:1b"

client = Client(host="http://localhost:11434")

PROMPT_PATTERNS = {
    "persona": {
        "template": (
            "You are {role}.\n"
            "Your communication style is {style}.\n"
            "You priopitize {priority}.\n\n"
            "{task}"
        ),
        "variables": [
            "role",
            "style",
            "priority",
            "task",
        ],
        "temperature": 0.5,
    },

    "decomposition": {
        "template": (
            "Problem:\n"
            "{problem}\n\n"
            "Break the problem into smaller sub-problems,"
            "solve them, and then produce a final answer."
        ),
        "variables": [
            "problem",
        ],
        "temperature": 0.3,
    }
}

def build_prompt(
        pattern_name: str,
        variables: dict[str, str],
        system_override: str | None = None,
) -> dict[str, Any]:

    if pattern_name not in PROMPT_PATTERNS:
        raise ValueError(
            f"Unknown pattern: {pattern_name}"
        )

    pattern = PROMPT_PATTERNS[pattern_name]

    missing = [
        variable
        for variable in pattern["variables"]
        if variable not in variables
    ]

    if missing:
        raise ValueError(
            f"Missing variables: {missing}"
        )

    user_prompt = pattern["template"].format(
        **variables
    )

    system_prompt = (
        system_override or "You are a precise AI engineering assistant."
    )

    return {
        "system": system_prompt,
        "user": user_prompt,
        "temperature": pattern["temperature"],
        "pattern": pattern_name,
    }

def call_llm(
        prompt: dict[str, Any],
        model: str = MODEL,
) -> dict[str, Any]:

    messages = [
        {"role": "system",
         "content": prompt["system"],
        },
        {
            "role": "user",
            "content": prompt["user"],
        },
    ]

    start = time.perf_counter()

    response = client.chat(
        model=model,
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
        "model": model,
    }

def run_prompt_test(
        pattern_name: str,
        variables: dict[str, str],
) -> dict[str, Any]:

    prompt = build_prompt(
        pattern_name,
        variables,
    )

    result = call_llm(prompt)

    return {
        "prompt": prompt,
        "result": result,
    }

def main() -> None:

    experiment = run_prompt_test(
        pattern_name="persona",
        variables={
            "role": "a senior AI engineer",
            "style": "precise and example-driven",
            "priority": "deep conceptual understanding",
            "task": "Explain why cosine similarity is useful for embedding retrival."
        },
    )

    print("\n--- PROMPT ---")
    print(experiment["prompt"]["user"])

    print("\n--- RESPONSE ---")
    print(
        experiment["result"]["response"]
    )

    print("\n--- METADATA ---")
    print(
        f"Model: "
        f"{experiment["result"]["model"]}"
    )

    print(
        f"Latancy: "
        f"{experiment["result"]["latency_ms"]} ms"
    )

if __name__ == "__main__":
    main()