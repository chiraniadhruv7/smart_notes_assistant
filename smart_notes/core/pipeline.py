from typing import List
from smart_notes.core.interfaces import (
    DocumentLoader,
    Chunker,
    Embedder,
    VectorStore,
    LLM,
)
from smart_notes.core.types import Document


class RAGPipeline:
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        llm: LLM,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def ingest(self, source: str):
        documents: List[Document] = self.loader.load(source)

        for doc in documents:
            chunks = self.chunker.chunk(doc.text)
            embeddings = self.embedder.embed(chunks)

            metadata = [{"source": source} for _ in chunks]
            self.vector_store.add(embeddings, metadata)

    def query(self, question: str, top_k: int = 5) -> str:
        query_embedding = self.embedder.embed([question])[0]
        results = self.vector_store.search(query_embedding, top_k)

        context = "\n".join([r.text for r in results])
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        return self.llm.generate(prompt)
