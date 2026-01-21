import os

from smart_notes.ingestion.document_loader import TextFileLoader
from smart_notes.ingestion.pdf_loader import PDFLoader
from smart_notes.core.interfaces import DocumentLoader


class LoaderFactory:
    @staticmethod
    def get_loader(source: str) -> DocumentLoader:
        ext = os.path.splitext(source)[1].lower()

        if ext == ".pdf":
            return PDFLoader()
        elif ext in {".txt", ".md"}:
            return TextFileLoader()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
