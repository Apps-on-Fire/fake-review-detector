import os
from pinecone import Pinecone
from openai import OpenAI

INDEX_NAME = "fake-reviews"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=os.environ["PINECODE_API_KEY"])


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def embed_text(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding


def embed_texts(client: OpenAI, texts: list[str], batch_size: int = 100) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        all_embeddings.extend([d.embedding for d in response.data])
    return all_embeddings


def query_similar(text: str, category: str | None = None, top_k: int = 5) -> list[dict]:
    pc = get_pinecone_client()
    oai = get_openai_client()

    index = pc.Index(INDEX_NAME)
    embedding = embed_text(oai, text)

    filter_dict = {"split": "train"}
    if category:
        filter_dict["category"] = category

    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict,
    )

    similar = []
    for match in results.matches:
        meta = match.metadata or {}
        similar.append({
            "review_id": match.id,
            "score": round(float(match.score), 4),
            "label": meta.get("label", ""),
            "category": meta.get("category", ""),
            "rating": meta.get("rating", 0),
            "text": meta.get("text", ""),
        })
    return similar
