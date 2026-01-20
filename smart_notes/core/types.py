from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Document:
    text: str
    metadata: Dict[str, Any]


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: Dict[str, Any]
