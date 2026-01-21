from typing import List
import numpy as np
from smart_notes.core.interfaces import Embedder


class DummyEmbeddingModel(Embedder):
    """
    Placeholder embedder.
    Replace with SentenceTransformers / OpenAI later.
    """

    def embed(self, texts: List[str]):
        return np.random.rand(len(texts), 384)
