import pickle
import os

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import settings


class Retriever:
    """Завантажує готовий індекс (з ingest.py) і реалізує hybrid retrieval + reranking."""

    def __init__(self):
        index_path = os.path.join(settings.index_dir, "faiss.index")
        chunks_path = os.path.join(settings.index_dir, "chunks.pkl")

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(
                f"Індекс не знайдено в {settings.index_dir}/. Спочатку запусти: python ingest.py"
            )

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        self.embed_model = SentenceTransformer(settings.embedding_model)
        self.reranker = CrossEncoder(settings.reranker_model)

        # BM25 будується по тим самим чанкам, що й FAISS-індекс — той самий порядок
        tokenized_corpus = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _semantic_search(self, query: str, top_k: int) -> list[int]:
        """Повертає індекси top_k найближчих чанків за cosine similarity."""
        query_vec = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        _, indices = self.index.search(query_vec, top_k)
        return [int(i) for i in indices[0] if i != -1]

    def _bm25_search(self, query: str, top_k: int) -> list[int]:
        """Повертає індекси top_k чанків за BM25-скором."""
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [int(i) for i in top_indices]

    def _reciprocal_rank_fusion(self, ranked_lists: list[list[int]], k: int = 60) -> list[int]:
        """Об'єднує кілька ранжованих списків індексів у один через RRF.

        Кожен елемент отримує 1/(k + rank) балів з кожного списку, де він з'явився;
        бали підсумовуються. Це уникає проблеми несумісних шкал (BM25-скор і
        cosine similarity виміряні по-різному і напряму не додаються).
        """
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, idx in enumerate(ranked):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """Повний pipeline: semantic + BM25 → RRF fusion → cross-encoder reranking → top_k."""
        top_k = top_k or settings.knowledge_search_top_k
        candidate_pool = min(len(self.chunks), max(top_k * 4, 20))

        semantic_hits = self._semantic_search(query, candidate_pool)
        bm25_hits = self._bm25_search(query, candidate_pool)

        fused_indices = self._reciprocal_rank_fusion([semantic_hits, bm25_hits])[:candidate_pool]

        candidates = [self.chunks[i] for i in fused_indices]
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]