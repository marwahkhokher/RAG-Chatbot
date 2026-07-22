from functools import lru_cache

from groq import Groq
from sentence_transformers import SentenceTransformer

from app import config
from app import qdrant_rest as qdrant

SYSTEM_PROMPT = """You are a friendly, helpful assistant having a conversation with a user.
You may be given some reference context pulled from a document collection. Use it when it's
relevant to the question. Do NOT refuse to answer just because a question isn't covered by
the context - greetings, small talk, and general knowledge questions should be answered
naturally using what you already know. Only say you don't know if you genuinely cannot
answer the question at all, even with general knowledge."""

MAX_HISTORY_MESSAGES = 10  # keep the prompt from growing without bound


@lru_cache
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def upsert_documents(chunks: list[str], sources: list[str]) -> None:
    vectors = embed_texts(chunks)
    qdrant.ensure_collection(vector_size=len(vectors[0]))
    payloads = [{"text": chunk, "source": source} for chunk, source in zip(chunks, sources)]
    qdrant.upsert_points(vectors, payloads)


@lru_cache
def get_llm_client() -> Groq:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=config.GROQ_API_KEY)


def answer_question(question: str, history: list[dict] | None = None, k: int = 4) -> dict:
    history = (history or [])[-MAX_HISTORY_MESSAGES:]

    query_vector = embed_query(question)
    qdrant.ensure_collection(vector_size=len(query_vector))
    results = qdrant.search(query_vector, k=k)
    context = "\n\n".join(r["payload"]["text"] for r in results)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]

    if context.strip():
        user_content = (
            f"Reference context (use only if relevant):\n{context}\n\nQuestion: {question}"
        )
    else:
        user_content = question
    messages.append({"role": "user", "content": user_content})

    client = get_llm_client()
    completion = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=0.4,
    )
    answer = completion.choices[0].message.content

    sources = [
        {
            "content": r["payload"]["text"][:200],
            "source": r["payload"].get("source", "unknown"),
        }
        for r in results
    ]
    return {"answer": answer, "sources": sources}