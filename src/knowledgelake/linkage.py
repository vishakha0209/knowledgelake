"""Stage 4b - Semantic duplicate detection, treated as a record-linkage problem.

Lexical matching (dedupe.py) is precise but misses paraphrases such as
"Explain the Medallion architecture" vs "Purpose of bronze, silver and gold layers".
Plain embedding similarity finds those, but it also merges the many SQL questions
that look alike and differ in one detail. So this stage works like entity resolution:

1. **Block** - candidate pairs are each question's k nearest neighbours in embedding
   space (plus the lexical candidates), instead of all n^2 pairs.
2. **Score** - a gradient-boosted model looks at 11 features per pair: question
   similarity, *answer* similarity, token Jaccard, character similarity,
   meaning-critical word agreement, length ratio, ... and outputs P(duplicate).
   It was trained on the pairs merged or rejected in a manual review of my notes
   (tools/train_pair_model.py); the shipped model holds only these numeric features.
3. **Decide** - p >= AUTO merges automatically; REVIEW <= p < AUTO goes to a
   ranked review queue (CSV) that a human confirms with ``kb review``.
"""
from __future__ import annotations

import csv
import difflib
import logging
from pathlib import Path

import numpy as np

from .dedupe import critical, tokens
from .embed import get_embedder, item_text
from .models import QAItem

log = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "data" / "pair_model.joblib"
AUTO, REVIEW = 0.8, 0.4
FEATURES = ["q_cosine", "a_cosine", "token_jaccard", "char_ratio", "critical_equal", "critical_diff",
            "length_ratio", "both_answered", "code_answers", "min_tokens", "same_source"]


class PairScorer:
    """Computes features for candidate pairs over one list of items."""

    def __init__(self, items: list[QAItem], k: int = 15, embedder=None):
        self.items = items
        self.q = [i.q for i in items]
        self.a = [i.a for i in items]
        emb = embedder or get_embedder()
        emb.fit([item_text(i.q, i.a) for i in items])
        self.EQ = emb.encode(self.q)
        emb_a = embedder or get_embedder()
        texts_a = [(i.a or i.q)[:1500] for i in items]
        emb_a.fit(texts_a)
        self.EA = emb_a.encode(texts_a)
        self.toks = [set(tokens(q)) for q in self.q]
        self.crit = [critical(tokens(q)) for q in self.q]
        self.k = k

    def candidates(self, focus: set[int] | None = None, extra: set[tuple[int, int]] | None = None):
        """k-nearest-neighbour blocking. With focus, only pairs touching those items."""
        rows = sorted(focus) if focus is not None else range(len(self.items))
        out = set(extra or ())
        k = min(self.k, len(self.items) - 1)
        if k <= 0:
            return []
        for i in rows:
            sims = self.EQ @ self.EQ[i]
            sims[i] = -1
            for j in np.argpartition(-sims, k - 1)[:k]:
                out.add((min(i, int(j)), max(i, int(j))))
        return sorted(out)

    def features(self, i: int, j: int) -> list[float]:
        A, B = self.toks[i], self.toks[j]
        ci, cj = self.crit[i], self.crit[j]
        si = self.items[i].sources[:1]
        return [float(self.EQ[i] @ self.EQ[j]), float(self.EA[i] @ self.EA[j]),
                len(A & B) / max(len(A | B), 1),
                difflib.SequenceMatcher(None, self.q[i].lower(), self.q[j].lower()).ratio(),
                float(ci == cj), float(len(ci ^ cj)),
                min(len(self.q[i]), len(self.q[j])) / max(len(self.q[i]), len(self.q[j]), 1),
                float(bool(self.a[i].strip()) and bool(self.a[j].strip())),
                float(self.a[i].lstrip().startswith("```")) + float(self.a[j].lstrip().startswith("```")),
                float(min(len(A), len(B))), float(bool(si) and si == self.items[j].sources[:1])]

    def matrix(self, pairs) -> np.ndarray:
        return np.array([self.features(i, j) for i, j in pairs], dtype=float).reshape(-1, len(FEATURES))


def load_model(path: Path = MODEL_PATH):
    import joblib
    return joblib.load(path)


def score_pairs(items: list[QAItem], focus: set[int] | None = None, model=None, embedder=None):
    """Return [(p, i, j)] for candidate pairs, highest probability first."""
    if len(items) < 2:
        return []
    if model is None:
        if not MODEL_PATH.exists():
            log.warning("no pair model at %s; semantic de-duplication skipped", MODEL_PATH)
            return []
        model = load_model()
    sc = PairScorer(items, embedder=embedder)
    pairs = sc.candidates(focus)
    if not pairs:
        return []
    p = model.predict_proba(sc.matrix(pairs))[:, 1]
    return sorted(((float(pi), i, j) for pi, (i, j) in zip(p, pairs)), reverse=True)


def split_decisions(scored, auto: float = AUTO, review: float = REVIEW):
    merges = [(i, j) for p, i, j in scored if p >= auto]
    queue = [(p, i, j) for p, i, j in scored if review <= p < auto]
    return merges, queue


def write_queue(path: Path, queue, items: list[QAItem]) -> int:
    """Ranked CSV for a human: fill 'decision' with y (same question) or n, then run `kb review`."""
    path = Path(path)
    rows = []
    if path.exists():  # keep earlier undecided / decided rows, add new ones
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    seen = {(r["question_a"], r["question_b"]) for r in rows}
    for p, i, j in queue:
        a, b = items[i], items[j]
        if (a.q, b.q) in seen or (b.q, a.q) in seen:
            continue
        rows.append({"p_duplicate": f"{p:.2f}", "decision": "", "question_a": a.q, "question_b": b.q,
                     "topic_a": a.topic, "topic_b": b.topic})
    rows.sort(key=lambda r: (r["decision"] != "", -float(r["p_duplicate"])))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["p_duplicate", "decision", "question_a", "question_b", "topic_a", "topic_b"])
        w.writeheader()
        w.writerows(rows)
    return sum(1 for r in rows if not r["decision"])
