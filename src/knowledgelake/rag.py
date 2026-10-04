"""Retrieval-augmented "Ask my notes".

Retrieval is hybrid:
* keyword  - TF-IDF over question + answer (abbreviations expanded), good at exact terms
             such as function names (``dense_rank``, ``OPTIMIZE``);
* semantic - embeddings (sentence-transformers or LSA, see embed.py), good at paraphrases;
* the two rankings are combined with reciprocal rank fusion (RRF), which needs no score
  calibration between the two systems.

Generation: with ANTHROPIC_API_KEY set, Claude writes an answer grounded only in the
retrieved notes and cites them as [1], [2]...; without a key the top notes are returned
as-is, so the app is still useful offline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .embed import expand, get_embedder
from .models import QAItem

RRF_K = 60


@dataclass
class Hit:
    item: QAItem
    score: float
    rank_keyword: int | None = None
    rank_semantic: int | None = None


def _doc_text(i: QAItem) -> str:
    # question weighted x3 + start of the answer: best of 4 variants on tools/evaluate_retrieval.py
    return f"{i.q} {i.q} {i.q} {i.a[:300]}"


class Retriever:
    def __init__(self, items: list[QAItem], embedder=None, mode: str = "hybrid"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.items, self.mode = items, mode
        docs = [_doc_text(i) for i in items]
        self.tfidf = TfidfVectorizer(preprocessor=expand, ngram_range=(1, 2), sublinear_tf=True,
                                     stop_words="english", min_df=1)
        self.K = self.tfidf.fit_transform(docs)
        self.emb = embedder or get_embedder()
        self.emb.fit(docs)
        self.E = self.emb.encode(docs)

    def _rank(self, scores: np.ndarray, allowed: np.ndarray, depth: int) -> list[int]:
        scores = np.where(allowed, scores, -np.inf)
        top = np.argpartition(-scores, min(depth, len(scores) - 1))[:depth]
        return [int(i) for i in top[np.argsort(-scores[top])] if np.isfinite(scores[i]) and scores[i] > 0]

    def search(self, query: str, k: int = 5, topic: str | None = None, depth: int = 50) -> list[Hit]:
        allowed = np.array([topic in (None, "", i.topic) for i in self.items])
        kw = self._rank((self.K @ self.tfidf.transform([query]).T).toarray().ravel(), allowed, depth)
        se = self._rank(self.E @ self.emb.encode([query])[0], allowed, depth)
        if self.mode == "keyword":
            se = []
        elif self.mode == "semantic":
            kw = []
        fused: dict[int, Hit] = {}
        for rank, i in enumerate(kw, 1):
            fused.setdefault(i, Hit(self.items[i], 0.0)).score += 1 / (RRF_K + rank)
            fused[i].rank_keyword = rank
        for rank, i in enumerate(se, 1):
            fused.setdefault(i, Hit(self.items[i], 0.0)).score += 1 / (RRF_K + rank)
            fused[i].rank_semantic = rank
        return sorted(fused.values(), key=lambda h: -h.score)[:k]


PROMPT = """You are helping a data engineer, using ONLY their own knowledge base below.

Notes:
{context}

Question: {question}

Answer in 80-200 words: the direct answer first, then key points or a short
code/SQL example if it helps. Cite the notes you used like [1] or [2]. If the notes don't cover
the question, say so in one sentence and give only a brief general answer, marked "(not in your notes)"."""


def answer(question: str, hits: list[Hit], api_key: str | None = None, model: str | None = None) -> dict:
    """Returns {"answer": str|None, "generated": bool}."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not hits:
        return {"answer": None, "generated": False}
    from .answer import DEFAULT_MODEL, _call
    context = "\n\n".join(f"[{n}] Q: {h.item.q}\nA: {h.item.a[:1500] or '(no answer in notes)'}"
                          for n, h in enumerate(hits, 1))
    text = _call(PROMPT.format(context=context, question=question), api_key, model or DEFAULT_MODEL)
    return {"answer": text or None, "generated": bool(text)}
