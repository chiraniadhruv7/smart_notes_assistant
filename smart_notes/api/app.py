from fastapi import FastAPI
from pydantic import BaseModel

from smart_notes.core.pipeline import RAGPipeline
from smart_notes.preprocessing.chunking import FixedSizeChunker
from smart_notes.embeddings.embedding_model import SentenceTransformerEmbedder
from smart_notes.vectorstore.faiss_store import FAISSVectorStore
from smart_notes.llm.generator import DummyLLM

app = FastAPI(title="Smart Notes Assistant")

embedder = SentenceTransformerEmbedder()
vector_store = FAISSVectorStore(dim=384)

pipeline = RAGPipeline(
    chunker=FixedSizeChunker(),
    embedder=embedder,
    vector_store=vector_store,
    llm=DummyLLM(),
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/ingest")
def ingest(source: str):
    pipeline.ingest(source)
    return {"status": "ingested", "source": source}


@app.post("/query")
def query(req: QueryRequest):
    answer = pipeline.query(req.question, req.top_k)
    return {"answer": answer}
