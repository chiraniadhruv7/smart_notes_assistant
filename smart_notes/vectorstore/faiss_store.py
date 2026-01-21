import faiss
import numpy as np
from smart_notes.core.interfaces import VectorStore
from smart_notes.core.types import SearchResult


class FAISSVectorStore(VectorStore):
    def __init__(self, dim: int = 384):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.metadata = []

    def add(self, embeddings, metadata):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding, top_k: int):
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append(
                SearchResult(
                    text=self.texts[idx] if idx < len(self.texts) else "",
                    score=float(dist),
                    metadata=self.metadata[idx]
                )
            )
        return results
