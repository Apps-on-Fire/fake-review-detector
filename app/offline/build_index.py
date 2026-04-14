"""
Script offline para indexar os reviews de treino no Pinecone.

Gera embeddings via OpenAI (text-embedding-3-small) e envia para o índice
Pinecone com metadados para filtragem.

Uso:
    python -m app.offline.build_index
"""

import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.retrieval.pinecone_client import (
    INDEX_NAME,
    EMBEDDING_DIM,
    get_pinecone_client,
    get_openai_client,
    embed_texts,
)
from app.ml.preprocessing import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "treinamento", "reviews_treino.csv")

BATCH_SIZE = 100


def ensure_index_exists():
    from pinecone import ServerlessSpec

    pc = get_pinecone_client()
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        print(f"Criando índice '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Aguardando índice ficar pronto...")
        while not pc.describe_index(INDEX_NAME).status.get("ready", False):
            time.sleep(2)
        print("Índice pronto.")
    else:
        print(f"Índice '{INDEX_NAME}' já existe.")


def main():
    print("Carregando dados de treino...")
    df = pd.read_csv(TRAIN_PATH)
    df = df.dropna(subset=["text_", "label"])
    df = df[df["label"].isin(["CG", "OR"])].copy()
    df["text_clean"] = df["text_"].apply(clean_text)
    df = df[df["text_clean"].str.strip().astype(bool)].copy()
    df = df.reset_index(drop=True)
    print(f"  {len(df)} reviews para indexar (após remover textos vazios)")

    ensure_index_exists()

    pc = get_pinecone_client()
    index = pc.Index(INDEX_NAME)
    oai = get_openai_client()

    total = len(df)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = df.iloc[start:end]

        texts = batch["text_clean"].tolist()
        embeddings = embed_texts(oai, texts, batch_size=BATCH_SIZE)

        vectors = []
        for i, (_, row) in enumerate(batch.iterrows()):
            review_id = f"train_{start + i:06d}"
            metadata = {
                "category": str(row["category"]),
                "rating": float(row["rating"]),
                "label": str(row["label"]),
                "split": "train",
                "text": str(row["text_"])[:500],
            }
            vectors.append({
                "id": review_id,
                "values": embeddings[i],
                "metadata": metadata,
            })

        index.upsert(vectors=vectors)
        progress = end / total * 100
        print(f"  Indexados {end}/{total} ({progress:.1f}%)")

    stats = index.describe_index_stats()
    print(f"\nIndexação completa. Total de vetores: {stats.total_vector_count}")


if __name__ == "__main__":
    main()
