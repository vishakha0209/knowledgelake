"""Automatic topic discovery: no hand-written topic list needed.

The built-in topic rules (config/presets/data_engineering.json) only make sense for
data-engineering documents. For anything else, this module finds the topics itself:

1. Embed every question + the start of its answer (embed.py: LSA, or sentence-transformers).
2. Cluster with k-means; k is chosen by silhouette score in a range scaled to the corpus size.
3. Name each cluster from its most distinctive terms (class-based TF-IDF: terms frequent in
   this cluster and rare in the others), e.g. "Vpn & Remote Access". With ANTHROPIC_API_KEY set,
   Claude can turn those terms into a short human name instead.
4. Large clusters are split again into sections the same way.

The output is a normal topics config (same format as the presets), so the keyword rules,
the learned classifier, the PDFs and the review queues all work unchanged. The config is
saved inside the knowledge base, so later batches are filed into the same topics. Rename or
merge topics by editing that JSON file.

On my 1,775 hand-labelled data-engineering questions, the discovered topics were about 70%
pure against my own 13 topics without seeing a single label (tools/evaluate_discovery.py).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter

import numpy as np

from .embed import ABBREVIATIONS, LSAEmbedder, get_embedder
from .models import QAItem

log = logging.getLogger(__name__)

# words that say what *kind* of question it is, not what it is about
QUESTION_WORDS = set("""what whats how why when where which who explain describe define difference differences
between vs versus compare comparison use used using uses example examples write query find get list show
does do did can could would should will make made way ways work works working mean means meaning
best good common commonly main key important need needs needed type types kind kinds give tell
handle handling approach implement implementation scenario case cases question answer answers yes no
like just also used one two new data""".split())
TOKEN = re.compile(r"(?u)\b[a-zA-Z][a-zA-Z0-9_+#.-]*[a-zA-Z0-9_+#]\b|\b[a-zA-Z]\b")


def _texts(items: list[QAItem]) -> list[str]:
    return [f"{i.q} {i.q} {i.a[:300]}" for i in items]


def _choose_k(E: np.ndarray, k_min: int, k_max: int, seed: int = 0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    scores = []
    for k in range(k_min, k_max + 1):
        km = KMeans(k, n_init=10, random_state=seed).fit(E)
        if len(set(km.labels_)) < 2:
            continue
        s = silhouette_score(E, km.labels_, metric="cosine",
                             sample_size=min(len(E), 2000), random_state=seed)
        scores.append((s, k, km.labels_))
    if not scores:
        return None
    top = max(s for s, _, _ in scores)
    # prefer fewer topics when the silhouette is practically tied: easier to read and rename
    return min((x for x in scores if x[0] >= top - 0.01), key=lambda x: x[1])


def _k_range(n: int, k: int | None) -> tuple[int, int]:
    if k:
        return k, k
    hi = max(2, min(20, round(math.sqrt(n))))
    lo = min(3, hi)
    return lo, hi


def distinctive_terms(texts: list[str], labels, top: int = 12) -> dict[int, list[tuple[str, float]]]:
    """Class-based TF-IDF: per cluster, terms that are frequent there and rare in other clusters."""
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    clusters = sorted(set(labels))
    docs = [" ".join(t for t, l in zip(texts, labels) if l == c) for c in clusters]
    cv = CountVectorizer(lowercase=True, token_pattern=TOKEN.pattern, ngram_range=(1, 2),
                         stop_words=list(ENGLISH_STOP_WORDS | QUESTION_WORDS), min_df=1)
    X = cv.fit_transform(docs).toarray().astype(float)
    vocab = np.array(cv.get_feature_names_out())
    tf = X / np.maximum(X.sum(1, keepdims=True), 1)
    avg = X.sum() / len(clusters)
    idf = np.log(1 + avg / np.maximum(X.sum(0), 1))
    score = tf * idf
    out = {}
    for row, c in enumerate(clusters):
        order = np.argsort(-score[row])
        terms, seen = [], set()
        for j in order:
            term = vocab[j]
            if score[row, j] <= 0 or len(term) < 2:
                break
            words = term.split()
            if any(w in seen for w in words) and len(words) == 1:
                continue
            terms.append((term, float(score[row, j])))
            seen.update(words)
            if len(terms) >= top:
                break
        out[c] = terms
    return out


ACRONYMS = set(ABBREVIATIONS) | set("sql api aws gcp vpn hr it pdf ai ml dbt sso mfa kpi faq ui ux id url csv json xml ssh etl sla sap crm erp gdpr hipaa okr".split())
SQL_WORDS = set("select from where join left right inner group order by having count sum avg max min case when then else end as on and or not null distinct limit insert update delete table".split())


def _nameable(term: str) -> bool:
    """Code-like terms (e.customer_id, c.id) and bare SQL keywords make poor topic names."""
    return not any(ch in term for ch in "._()") and not all(w in SQL_WORDS for w in term.split())


def _name(terms: list[tuple[str, float]], taken: set[str]) -> str:
    words, used = [], set()
    for t, _ in terms:
        parts = set(t.split())
        if _nameable(t) and not parts & used:   # "Password & Reset Password" -> "Password & VPN"
            words.append(t)
            used |= parts
        if len(words) == 4:
            break
    if not words:
        return "General"
    def cap(term):
        return " ".join(w.upper() if w in ACRONYMS else w.capitalize() for w in term.split())
    name = " & ".join(cap(w) for w in words[:2])
    n = 2
    while name in taken and n < len(words):
        name = " & ".join(cap(w) for w in words[:2]) + f" ({cap(words[n])})"
        n += 1
    return name


def _llm_names(groups: dict[int, dict], api_key: str) -> dict[int, str]:
    """Optional: ask Claude for short names. Falls back silently to term-based names."""
    from .answer import DEFAULT_MODEL, _call
    lines = []
    for c, g in groups.items():
        lines.append(f"{c}: terms = {', '.join(t for t, _ in g['terms'][:8])}; examples = "
                     + " | ".join(q[:90] for q in g["examples"][:5]))
    prompt = ("Name each cluster of questions with a short topic name (2-4 words, Title Case, no numbering). "
              "Reply with only a JSON object mapping cluster id to name.\n\n" + "\n".join(lines))
    text = _call(prompt, api_key, DEFAULT_MODEL)
    try:
        m = re.search(r"\{.*\}", text, re.S)
        names = {int(k): str(v).strip() for k, v in json.loads(m.group(0)).items()}
        return {c: n for c, n in names.items() if n}
    except Exception:  # noqa: BLE001 - any parsing problem means "use the term-based names"
        return {}


def _weights(terms: list[tuple[str, float]], n: int) -> dict[str, float]:
    terms = terms[:n]
    if not terms:
        return {}
    top = terms[0][1]
    return {t: round(1 + 2 * s / top, 2) for t, s in terms}


def discover_topics(items: list[QAItem], k: int | None = None, sections: bool = True,
                    embedder=None, seed: int = 0, use_llm: bool | None = None) -> tuple[dict, list[str], list[str]]:
    """Returns (config, topic per item, section per item)."""
    n = len(items)
    if n < 6:
        cfg = {"fallback_topic": "General", "discovered": True, "topics": {"General": {"keywords": {}, "sections": {}}}}
        return cfg, ["General"] * n, ["General"] * n
    texts = _texts(items)
    emb = embedder or get_embedder()
    if isinstance(emb, LSAEmbedder):
        emb.dim = min(emb.dim, 100)  # fewer dimensions cluster better (tools/evaluate_discovery.py)
    E = emb.fit(texts).encode(texts)
    lo, hi = _k_range(n, k)
    _, k_used, labels = _choose_k(E, lo, hi, seed)
    terms = distinctive_terms(texts, labels)

    groups = {c: {"terms": terms[c], "idx": [i for i, l in enumerate(labels) if l == c]} for c in sorted(set(labels))}
    for g in groups.values():
        g["examples"] = [items[i].q for i in g["idx"][:5]]
    key = os.environ.get("ANTHROPIC_API_KEY") if use_llm in (None, True) else None
    llm = _llm_names(groups, key) if key else {}

    cfg = {"fallback_topic": "General", "discovered": True, "section_order": {}, "topics": {}}
    topic_of, section_of = [""] * n, ["General"] * n
    taken: set[str] = set()
    for c, g in sorted(groups.items(), key=lambda kv: -len(kv[1]["idx"])):
        name = llm.get(c) or _name(g["terms"], taken)
        if name in taken:
            name = f"{name} {c + 1}"
        taken.add(name)
        spec = {"keywords": _weights(g["terms"], 12), "sections": {}}
        for i in g["idx"]:
            topic_of[i] = name
        if sections and len(g["idx"]) >= 40:
            sub_items = [items[i] for i in g["idx"]]
            sub_texts = _texts(sub_items)
            sub_E = E[g["idx"]]
            best = _choose_k(sub_E, 2, min(6, max(2, len(sub_items) // 20)), seed)
            if best:
                _, _, sub_labels = best
                sub_terms = distinctive_terms(sub_texts, sub_labels, top=8)
                sec_taken: set[str] = set()
                order = []
                for sc in sorted(set(sub_labels), key=lambda s: -list(sub_labels).count(s)):
                    sname = _name(sub_terms[sc], sec_taken)
                    sec_taken.add(sname)
                    spec["sections"][sname] = _weights(sub_terms[sc], 8)
                    order.append(sname)
                    for local, l in enumerate(sub_labels):
                        if l == sc:
                            section_of[g["idx"][local]] = sname
                cfg["section_order"][name] = order
        cfg["topics"][name] = spec
    log.info("discovered %d topics (k chosen from %d-%d)", k_used, lo, hi)
    return cfg, topic_of, section_of


def summarise(cfg: dict, counts: dict[str, int] | None = None) -> str:
    lines = []
    for name, spec in cfg["topics"].items():
        kw = ", ".join(list(spec.get("keywords", {}))[:6])
        n = f"{counts.get(name, 0):5d}  " if counts is not None else ""
        lines.append(f"{n}{name}   [{kw}]")
        for s in spec.get("sections", {}):
            lines.append(f"{'':7s}  └ {s}" if counts is not None else f"  └ {s}")
    return "\n".join(lines)
