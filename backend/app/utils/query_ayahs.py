import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from typing import List
from ..models.query import AyahResult


load_dotenv()

embedding_client = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
);


def get_embedding(text):
    try:
        embedding = embedding_client.embed_query(text)
        return list(np.array(embedding, dtype=np.float32))
    except Exception as e:
        print("Error while getting embedding:", e)
        return None



def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=False,
        timeout=30.0
    )


def search_ayahs(query_vector: List[float], limit: int = 15) -> List[AyahResult]:
    client = get_qdrant_client()
    search_response = client.search(
        collection_name="quran_embeddings",
        query_vector=query_vector,
        limit=limit
    )
    return [
        AyahResult(
            score=hit.score,
            english_translation=hit.payload["english_translation"],
            surah_name_english=hit.payload["surah_name_english"],
            aya_number=hit.payload["aya_number"],
            arabic_diacritics=hit.payload.get("arabic_diacritics", "")
        )
        for hit in search_response
    ]

