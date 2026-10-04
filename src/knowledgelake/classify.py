"""Stage 3 - Assign every item to a topic and a section using weighted keywords.

Rules live in a topics config (a preset, or one discovered automatically) so topics change without code
changes. Each topic has keyword weights; the highest score wins. Sections work
the same way inside a topic. Question text counts double compared to the answer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .models import QAItem

PRESETS = Path(__file__).resolve().parent / "presets"
DEFAULT_CONFIG = PRESETS / "data_engineering.json"


def preset_names() -> list[str]:
    return sorted(p.stem.replace("_", "-") for p in PRESETS.glob("*.json"))


def load_config(path: Path | str | None = None) -> dict:
    """A JSON path, a preset name ("data-engineering"), or None for the data-engineering preset."""
    if path is None:
        return json.loads(DEFAULT_CONFIG.read_text())
    preset = PRESETS / f"{str(path).replace('-', '_')}.json"
    if not Path(path).exists() and preset.exists():
        return json.loads(preset.read_text())
    return json.loads(Path(path).read_text())


def _score(text: str, keywords: dict[str, float]) -> float:
    total = 0.0
    for kw, w in keywords.items():
        hits = len(re.findall(r"(?<![a-z0-9_])" + re.escape(kw.lower()) + r"(?![a-z0-9_])", text))
        total += hits * w
    return total


def topic_scores(item: QAItem, cfg: dict) -> dict[str, float]:
    q, a = item.q.lower(), item.a.lower()[:1500]
    return {t: 2 * _score(q, spec["keywords"]) + _score(a, spec["keywords"]) for t, spec in cfg["topics"].items()}


def classify(item: QAItem, cfg: dict, prior: dict[str, float] | None = None) -> QAItem:
    """prior = small bonus per topic from the document the item came from (document context)."""
    scores = topic_scores(item, cfg)
    if prior:
        for t, b in prior.items():
            scores[t] = scores.get(t, 0) + b
    best, best_score = cfg.get("fallback_topic", "Uncategorised"), 0.0
    for topic, s in scores.items():
        if s > best_score:
            best, best_score = topic, s
    item.topic = best
    _section(item, cfg)
    return item


def _section(item: QAItem, cfg: dict) -> None:
    sections = cfg["topics"].get(item.topic, {}).get("sections", {})
    q, a = item.q.lower(), item.a.lower()[:1500]
    sec, sec_score = "General", 0.0
    for name, kws in sections.items():
        s = 2 * _score(q, kws) + _score(a, kws)
        if s > sec_score:
            sec, sec_score = name, s
    if item.a.lstrip().startswith("```") and "practice_section" in cfg["topics"].get(item.topic, {}):
        sec = cfg["topics"][item.topic]["practice_section"]
    item.section = sec


class TopicModel:
    """Logistic regression on TF-IDF (word + character n-grams), trained on the knowledge base itself.

    Every topic already in the KB is a label, so each new batch is classified by a model
    trained on everything filed so far. Its probabilities are blended with the keyword rules
    (which carry domain knowledge the model can't learn from ~2k examples) and with the
    document-context prior. Under leave-source-file-out cross-validation on my notes this
    moved accuracy from 83.4% (rules) to 85.2%, and - more usefully - gives a confidence
    score: the least confident 20% of items hold two thirds of the errors.
    """
    MIN_ITEMS = 200

    def __init__(self, C: float = 2.0):
        self.C = C

    @staticmethod
    def _text(i: QAItem) -> str:
        return f"{i.q} || {i.q} || {i.a[:1500]}"

    def fit(self, items: list[QAItem]) -> "TopicModel":
        from scipy.sparse import hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        texts = [self._text(i) for i in items]
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2,
                                    max_features=150000)
        X = hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        self.clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced").fit(X, [i.topic for i in items])
        self.classes_ = list(self.clf.classes_)
        return self

    def predict_proba(self, items: list[QAItem]):
        from scipy.sparse import hstack
        texts = [self._text(i) for i in items]
        X = hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()
        return self.clf.predict_proba(X)

    @classmethod
    def from_kb(cls, kb_items: list[QAItem]) -> "TopicModel | None":
        labelled = [i for i in kb_items if i.topic and i.topic != "Uncategorised"]
        if len(labelled) < cls.MIN_ITEMS or len({i.topic for i in labelled}) < 2:
            return None
        return cls().fit(labelled)


def classify_all_learned(items: list[QAItem], cfg: dict, model: TopicModel,
                         w_rules: float = 0.6, w_doc: float = 1.0) -> list[QAItem]:
    """Blend learned probabilities with normalised keyword scores, then add the document prior.
    Sets item.topic_confidence = (best - runner-up) / total, in [0, 1]."""
    import numpy as np
    from collections import defaultdict
    if not items:
        return items
    topics = sorted(set(cfg["topics"]) | set(model.classes_))
    P_model = model.predict_proba(items)
    P = np.zeros((len(items), len(topics)))
    for k, t in enumerate(model.classes_):
        P[:, topics.index(t)] = P_model[:, k]
    scores = [topic_scores(i, cfg) for i in items]
    R = np.array([[sc.get(t, 0.0) for t in topics] for sc in scores])
    R = R / np.maximum(R.sum(1, keepdims=True), 1e-9)
    P = (1 - w_rules) * P + w_rules * R
    by_doc = defaultdict(list)
    for k, i in enumerate(items):
        by_doc[i.sources[0] if i.sources else ""].append(k)
    prior = np.zeros_like(P)
    for ks in by_doc.values():
        prior[ks] = P[ks].mean(0)
    P = P + w_doc * prior
    Ps = np.sort(P, 1)
    conf = (Ps[:, -1] - Ps[:, -2]) / np.maximum(Ps.sum(1), 1e-9)
    for k, i in enumerate(items):
        i.topic = topics[int(P[k].argmax())]
        i.topic_confidence = round(float(conf[k]), 3)
        _section(i, cfg)
    return items


def classify_all(items: list[QAItem], cfg: dict, doc_weight: float = 2.0) -> list[QAItem]:
    """Two passes: score items alone, then give each item a bonus for its document's dominant topic.
    A question like "How would you implement an incremental load?" is ambiguous on its own but
    clearly ADF when it sits in a document full of ADF questions."""
    from collections import Counter, defaultdict
    votes: dict[str, Counter] = defaultdict(Counter)
    for i in items:
        s = topic_scores(i, cfg)
        top = max(s, key=s.get)
        if s[top] > 0:
            votes[i.sources[0] if i.sources else ""][top] += 1
    for i in items:
        v = votes.get(i.sources[0] if i.sources else "", Counter())
        n = sum(v.values())
        prior = {t: doc_weight * c / n for t, c in v.items()} if n else None
        classify(i, cfg, prior)
    return items
