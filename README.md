# LLM and RAG Experiments

Small Python examples for working with local large language models (LLMs) through Ollama. RAG means retrieval-augmented generation: finding relevant text before asking a model to answer.

## Files

| File | Purpose |
| --- | --- |
| `prompt_patterns.py` | Builds persona and problem-decomposition prompts, runs a persona example, and reports response time. |
| `ollama_chat_client.py` | Provides a reusable `OllamaProvider` class for chat and text generation, with a runnable example. |
| `llm_response_timer.py` | Provides `call_llm()` to return an answer, elapsed time in milliseconds, and the model name. Import this helper into another script. |
| `retrieval_augmented_generation.py` | Embeds three sample documents, retrieves relevant text using cosine similarity, and generates an answer from that context. |

## Setup

Use Python 3.12 or newer and install Ollama. In this repository, create a Python environment and install the dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install ollama numpy
```

With Ollama running locally at `http://localhost:11434`, download the models used by these examples:

```sh
ollama pull gemma3:1b
ollama pull embeddinggemma:latest
```

## Run

```sh
python ollama_chat_client.py
python prompt_patterns.py
python retrieval_augmented_generation.py
```

To use the response timer in Python:

```python
from llm_response_timer import call_llm

result = call_llm({
    "system": "You are an AI engineering tutor.",
    "user": "Explain cosine similarity.",
    "temperature": 0.3,
})
print(result)
```
