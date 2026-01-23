import faiss
import numpy as np
from typing import List

from smart_notes.core.interfaces import VectorStore
from smart_notes.core.types import SearchResult


class FAISSVectorStore(VectorStore):
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.texts: List[str] = []
        self.metadata: List[dict] = []

    def add(self, embeddings, texts, metadata):
        embeddings = np.asarray(embeddings).astype("float32")

        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadata.extend(metadata)

    def search(self, query_embedding, top_k: int):
        query_embedding = np.asarray([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append(
                SearchResult(
                    text=self.texts[idx],
                    score=float(dist),
                    metadata=self.metadata[idx],
                )
            )
        return results
