from abc import ABC, abstractmethod
from typing import List


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, source: str):
        """Load raw documents from a source"""
        pass


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks"""
        pass


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]):
        """Convert text chunks into embeddings"""
        pass


class VectorStore(ABC):
    @abstractmethod
    def add(self, embeddings, metadata):
        """Store embeddings"""
        pass

    @abstractmethod
    def search(self, query_embedding, top_k: int):
        """Search similar embeddings"""
        pass


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from prompt"""
        pass
