import os
import pickle

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def load_pdf_pages(path: str) -> list[dict]:
    """Повертає список {"text", "source", "page"} — по одному запису на сторінку PDF."""
    reader = PdfReader(path)
    filename = os.path.basename(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "source": filename, "page": i})
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Розбиває текст кожної сторінки на чанки, зберігаючи метадані (source, page)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = []
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            chunks.append({"text": piece, "source": page["source"], "page": page["page"]})
    return chunks


def build_index():
    if not os.path.isdir(settings.data_dir):
        print(f"Директорія {settings.data_dir} не знайдена.")
        return

    pdf_files = [f for f in os.listdir(settings.data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"У {settings.data_dir} не знайдено жодного PDF файлу.")
        return

    print(f"Знайдено {len(pdf_files)} PDF файлів: {pdf_files}")

    all_pages = []
    for filename in pdf_files:
        pages = load_pdf_pages(os.path.join(settings.data_dir, filename))
        print(f"  {filename}: {len(pages)} сторінок з текстом")
        all_pages.extend(pages)

    chunks = chunk_pages(all_pages)
    print(f"Отримано {len(chunks)} чанків")

    print(f"Завантажуємо модель ембеддингів: {settings.embedding_model}...")
    model = SentenceTransformer(settings.embedding_model)

    texts = [c["text"] for c in chunks]
    print("Обчислюємо ембеддинги...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product на нормалізованих векторах = cosine similarity
    index.add(embeddings)

    os.makedirs(settings.index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(settings.index_dir, "faiss.index"))

    with open(os.path.join(settings.index_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"Індекс збережено в {settings.index_dir}/ ({len(chunks)} чанків, розмірність {dimension})")


if __name__ == "__main__":
    build_index()