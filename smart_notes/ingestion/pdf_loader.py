from typing import List
from pypdf import PdfReader

from smart_notes.core.interfaces import DocumentLoader
from smart_notes.core.types import Document


class PDFLoader(DocumentLoader):
    def load(self, source: str) -> List[Document]:
        reader = PdfReader(source)
        documents = []

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source": source,
                        "page": page_number,
                        "type": "pdf"
                    }
                )
            )

        return documents
