"""Text embeddings shared by de-duplication, classification and the "Ask my notes" chat.

Two interchangeable backends:

* ``sentence-transformers`` - a pretrained neural model (default ``all-MiniLM-L6-v2``).
  Used automatically when the package is installed and the model can be loaded.
* ``lsa`` - TF-IDF (word 1-2 grams + character 3-5 grams) reduced with truncated SVD
  (latent semantic analysis), fitted on your own notes. No downloads, no GPU, a few
  seconds for a few thousand questions. Domain abbreviations are expanded first, so
  "CDC" and "change data capture" land close together.

Pick one with ``KB_EMBED_BACKEND=lsa|sentence-transformers`` (default: auto).
Vectors are L2-normalised, so a dot product is the cosine similarity.
"""
from __future__ import annotations

import logging
import os
import re

import numpy as np

log = logging.getLogger(__name__)

ABBREVIATIONS = {
    "cdc": "change data capture", "scd": "slowly changing dimension", "etl": "extract transform load",
    "elt": "extract load transform", "adls": "azure data lake storage", "adf": "azure data factory",
    "ir": "integration runtime", "shir": "self hosted integration runtime", "rdd": "resilient distributed dataset",
    "dag": "directed acyclic graph", "udf": "user defined function", "cte": "common table expression",
    "rag": "retrieval augmented generation", "llm": "large language model",
    "oltp": "online transaction processing", "olap": "online analytical processing",
    "acid": "atomicity consistency isolation durability", "aqe": "adaptive query execution",
    "uc": "unity catalog", "dlt": "delta live tables", "rbac": "role based access control",
    "pk": "primary key", "fk": "foreign key", "medallion": "medallion bronze silver gold",
}
_WORD = re.compile(r"[a-z]+")


def expand(text: str) -> str:
    text = text.lower()
    return _WORD.sub(lambda m: f"{m.group(0)} {ABBREVIATIONS[m.group(0)]}" if m.group(0) in ABBREVIATIONS
                     else m.group(0), text)


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim, self.seed = dim, seed
        self._fitted = False

    def fit(self, corpus: list[str]) -> "LSAEmbedder":
        from scipy.sparse import hstack
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        min_df = 2 if len(corpus) > 50 else 1
        self.word = TfidfVectorizer(preprocessor=expand, ngram_range=(1, 2), sublinear_tf=True,
                                    min_df=min_df, stop_words="english")
        self.char = TfidfVectorizer(preprocessor=expand, analyzer="char_wb", ngram_range=(3, 5),
                                    sublinear_tf=True, min_df=min_df)
        X = hstack([self.word.fit_transform(corpus), self.char.fit_transform(corpus)]).tocsr()
        dim = max(2, min(self.dim, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(dim, random_state=self.seed).fit(X)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        from scipy.sparse import hstack
        if not self._fitted:
            self.fit(texts)
        X = hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()
        return _normalise(self.svd.transform(X))


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model: str | None = None):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.model_name = model or os.environ.get("KB_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)

    def fit(self, corpus: list[str]) -> "SentenceTransformerEmbedder":
        return self  # pretrained - nothing to fit

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.asarray(self.model.encode(texts, batch_size=64, show_progress_bar=False)))


def _normalise(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def get_embedder(backend: str | None = None):
    """Return a fresh embedder. 'auto' tries sentence-transformers and falls back to LSA."""
    backend = (backend or os.environ.get("KB_EMBED_BACKEND", "auto")).lower()
    if backend in ("auto", "sentence-transformers", "st"):
        try:
            return SentenceTransformerEmbedder()
        except Exception as e:  # not installed, or the model can't be downloaded
            if backend != "auto":
                raise
            log.info("sentence-transformers unavailable (%s); using LSA embeddings", e.__class__.__name__)
    return LSAEmbedder()


def item_text(q: str, a: str, answer_chars: int = 600) -> str:
    """Text used to fit the embedding space: the question plus the start of its answer."""
    return f"{q} {a[:answer_chars]}"
