from __future__ import annotations

import numpy as np
import ollama

EMBEDDING_MODEL = "embeddinggemma:latest"

LLM_MODEL = "gemma3:1b"

documents = [
    (
        "RAG stands for Retrieval-Augmented "
        "Generation. It retrieves documents "
        "before asking the LLM to answer."
    ),

    (
        "Embeddings convert text into vectors "
        "that capture semantic meaning."
    ),

    (
        "Cosine similarity measures the angle "
        "between two vectors and is commonly "
        "used for embedding retrieval."
    ),
]


def normalize(
    matrix: np.ndarray,
) -> np.ndarray:
    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    return matrix / np.clip(
        norms,
        1e-12,
        None,
    )

def embed_texts(texts: list[str],) -> np.ndarray:
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    matrix = np.asarray(
        response.embeddings,
        dtype=np.float32,
    )

    return normalize(matrix)

def retrieve(query: str, document_embeddings: np.ndarray, top_k: int = 2,) -> list[str]:

    query_embedding = embed_texts([query])[0]

    scores = document_embeddings @ query_embedding

    indices = np.argsort(scores)[::-1][:top_k]

    return [documents[index] for index in indices]

def generate_answer(question: str, contexts: list[str],) -> str:

    context_text = "\n\n".join(contexts)

    prompt = f"""
Answer the question using only the provided context.

Context:
{context_text}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model=LLM_MODEL,

        messages=[
            {
                "role":"user",
                "content": prompt,
            }
        ],

        options={
            "temperature": 0.2,
        },
    )

    return (
        response.message.content or ""
    )

def main() -> None:
    document_embeddings = embed_texts(documents)

    question = "Why is cosine similarity useful for semantic search?"

    contexts = retrieve(
        question,
        document_embeddings,
        top_k=2,
    )

    print("--- RETRIEVED ---")
    for context in contexts:
        print(context)

    answer = generate_answer(question, contexts,)

    print("\n--- ANSWER ---")
    print(answer)

if __name__ == "__main__":
    main()

    