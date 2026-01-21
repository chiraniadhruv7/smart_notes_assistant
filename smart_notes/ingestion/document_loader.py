from typing import List
from smart_notes.core.interfaces import DocumentLoader
from smart_notes.core.types import Document


class TextFileLoader(DocumentLoader):
    def load(self, source: str) -> List[Document]:
        with open(source, "r", encoding="utf-8") as f:
            text = f.read()

        return [
            Document(
                text=text,
                metadata={"source": source}
            )
        ]
