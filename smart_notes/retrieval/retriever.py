from typing import List

from smart_notes.core.interfaces import Embedder, VectorStore
from smart_notes.core.types import SearchResult


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[SearchResult]:
        query_embedding = self.embedder.embed([query])[0]
        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )
