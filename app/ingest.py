"""
Run this once (and again whenever you add new documents) to populate Qdrant.

Usage:
    python -m app.ingest
"""
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.rag import upsert_documents

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_documents() -> list[tuple[str, str]]:
    """Returns a list of (text, source_filename) tuples."""
    docs = []
    for path in DATA_DIR.glob("**/*"):
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            docs.append((text, path.name))
        elif path.suffix.lower() in (".txt", ".md"):
            docs.append((path.read_text(encoding="utf-8"), path.name))
    return docs


def main():
    docs = load_documents()
    if not docs:
        print(f"No documents found in {DATA_DIR}. Add .txt, .md, or .pdf files there first.")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks: list[str] = []
    sources: list[str] = []
    for text, source in docs:
        for chunk in splitter.split_text(text):
            chunks.append(chunk)
            sources.append(source)

    upsert_documents(chunks, sources)
    print(f"Ingested {len(docs)} document(s) as {len(chunks)} chunks into Qdrant.")


if __name__ == "__main__":
    main()
