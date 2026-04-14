"""Testa conexão com OpenAI e Pinecone usando variáveis de ambiente."""
import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

errors = []

# Test OpenAI
key = os.environ.get("OPENAI_API_KEY", "")
if not key:
    errors.append("OPENAI_API_KEY não definida")
else:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.embeddings.create(input=["test"], model="text-embedding-3-small")
        print(f"OpenAI OK - embedding dim: {len(resp.data[0].embedding)}")
    except Exception as e:
        errors.append(f"OpenAI erro: {e}")

# Test Pinecone
key = os.environ.get("PINECODE_API_KEY", "")
if not key:
    errors.append("PINECODE_API_KEY não definida")
else:
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=key)
        indexes = [idx.name for idx in pc.list_indexes()]
        print(f"Pinecone OK - índices existentes: {indexes}")
    except Exception as e:
        errors.append(f"Pinecone erro: {e}")

if errors:
    print("\nERROS:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nTudo OK! Pode rodar a indexação.")
