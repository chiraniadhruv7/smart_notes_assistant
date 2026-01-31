FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy project metadata
COPY pyproject.toml .

# Install Python dependencies (EXPLICIT)
RUN pip install --upgrade pip \
    && pip install \
        fastapi \
        uvicorn \
        sentence-transformers \
        faiss-cpu \
        pypdf \
    && pip install -e .

# Copy source code
COPY smart_notes smart_notes

CMD ["uvicorn", "smart_notes.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
