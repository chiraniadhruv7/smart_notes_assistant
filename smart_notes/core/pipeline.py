from typing import List

from smart_notes.core.interfaces import (
    Chunker,
    Embedder,
    VectorStore,
    LLM,
)
from smart_notes.core.types import Document
from smart_notes.ingestion.factory import LoaderFactory
from smart_notes.retrieval.retriever import Retriever



class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation (RAG) pipeline.

    Responsibilities:
    - Ingest documents of multiple formats (txt, pdf, etc.)
    - Chunk and embed text
    - Store embeddings in a vector database
    - Retrieve relevant context and generate answers using an LLM
    """

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        llm: LLM,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def ingest(self, source: str) -> None:
        """
        Ingest a document source (txt, pdf, etc.) into the vector store.
        """

        loader = LoaderFactory.get_loader(source)
        documents: List[Document] = loader.load(source)

        for doc in documents:
            chunks = self.chunker.chunk(doc.text)

            embeddings = self.embedder.embed(chunks)

            metadatas = [
                {
                    **doc.metadata,
                    "chunk_id": idx,
                }
                for idx in range(len(chunks))
            ]

            self.vector_store.add(
                embeddings=embeddings,
                texts=chunks,
                metadata=metadatas,
            )

    def query(self, question: str, top_k: int = 5) -> str:
        retriever = Retriever(
        embedder=self.embedder,
        vector_store=self.vector_store,
        top_k=top_k,
    )

        results = retriever.retrieve(question)

        context = "\n\n".join(r.text for r in results)

        prompt = (
        "You are a helpful assistant.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

        return self.llm.generate(prompt)

