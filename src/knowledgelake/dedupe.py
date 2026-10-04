"""Stage 4 - Find and merge duplicate questions.

Two questions are duplicates when, after normalisation, their token sets are
very similar (Jaccard >= threshold) AND they share the same "critical" tokens.
Critical tokens are the words that flip the meaning of an otherwise identical
question - numbers, highest/lowest, first/last, never, before/after, left/right
join ... - so "second highest salary" is never merged with "second lowest salary".

Candidate pairs come from an inverted index (token -> question ids), so the
comparison is near-linear instead of O(n^2). Duplicate groups are resolved with
union-find, and each group keeps the richest answer; other answers are appended
only when they add genuinely new content.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .models import QAItem

STOP = set(
    "a an the of in on for to and or is are was what how do does you your would we with by from be it this "
    "that which when why can table tables who whose has have had been there their them its please explain "
    "describe define difference between vs versus".split()
    # technology qualifiers rarely change meaning ("broadcast join" == "broadcast join in Spark")
    + "spark pyspark sql databricks delta adf azure python snowflake".split()
)
SYNONYMS = {
    "hired": "join", "joined": "join", "salaries": "salary", "purchased": "order", "purchase": "order",
    "purchases": "order", "ordered": "order", "orders": "order", "bought": "order", "per": "each",
    "every": "each", "customers": "customer", "products": "product", "departments": "department",
    "employees": "employee", "dept": "department", "retrieve": "", "list": "", "show": "", "display": "",
    "identify": "", "calculate": "", "compute": "", "determine": "", "return": "", "fetch": "",
    "select": "", "query": "", "write": "", "get": "", "find": "", "made": "", "placed": "", "within": "",
}
CRITICAL = set(
    "highest lowest max min maximum minimum first last second third nth top bottom before after never both only "
    "all any most least once twice more less fewer same different without above below increase decrease "
    "consecutive month year day week quarter hour previous next current latest earliest oldest newest average "
    "median sum count total percentage running moving cumulative delete insert update left right inner outer "
    "full self cross semi anti union intersect except".split()
)


def tokens(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("&", " and ").replace("()", ""))
    out = []
    for t in text.split():
        if t in STOP:
            continue
        t = SYNONYMS.get(t, t)
        if not t:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def match_tokens(text: str) -> list[str]:
    """tokens() for matching; if every word is a stop word ("What is PySpark?") fall back to all words,
    otherwise such questions would have no tokens and never be recognised as duplicates."""
    return tokens(text) or re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def critical(toks: list[str]) -> frozenset[str]:
    return frozenset(t for t in toks if t in CRITICAL or t.isdigit())


def is_duplicate(q1: str, q2: str, threshold: float = 0.8) -> bool:
    a, b = match_tokens(q1), match_tokens(q2)
    A, B = set(a), set(b)
    if not A or not B or critical(a) != critical(b):
        return False
    return len(A & B) / len(A | B) >= threshold


def find_groups(items: list[QAItem], threshold: float = 0.8, max_df: int = 300,
                extra_pairs: list[tuple[int, int]] | None = None) -> list[list[int]]:
    """Lexical duplicate groups; extra_pairs (e.g. from the semantic stage) are unioned in too."""
    toks = [match_tokens(i.q) for i in items]
    sets = [set(t) for t in toks]
    crit = [critical(t) for t in toks]
    inv: dict[str, set[int]] = defaultdict(set)
    for idx, s in enumerate(sets):
        for t in s:
            inv[t].add(idx)

    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, s in enumerate(sets):
        if not s:
            continue
        shared = Counter()
        informative = [t for t in s if len(inv[t]) <= max_df]  # ultra-common tokens don't find candidates
        if not informative:  # ...unless that is all the question has: use its rarest token
            informative = [min(s, key=lambda t: len(inv[t]))]
        for t in informative:
            for j in inv[t]:
                if j > i:
                    shared[j] += 1
        for j in shared:
            if crit[i] != crit[j]:
                continue
            # frequent tokens are skipped only for *finding* candidates; similarity uses all tokens
            # (counting only rare tokens made identical questions full of common words score < threshold)
            if len(s & sets[j]) / len(s | sets[j]) >= threshold:
                parent[find(i)] = find(j)

    for i, j in extra_pairs or ():
        parent[find(i)] = find(j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    return list(groups.values())


def _content_tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{3,}", s.lower()))


def merge_group(group: list[QAItem], novelty: float = 0.5) -> QAItem:
    """Keep the best answer; append other answers only if >novelty of their words are new."""
    answered = [g for g in group if g.has_answer]
    primary = max(answered or group, key=lambda g: (not g.answered_by_llm, len(g.a)))
    seen = _content_tokens(primary.a)
    for g in group:
        for s in g.sources:
            if s not in primary.sources:
                primary.sources.append(s)
        if g is primary or not g.has_answer:
            continue
        t = _content_tokens(g.a)
        if len(g.a) > 80 and t and len(t - seen) / len(t) > novelty:
            label = "Alternative solution" if g.a.lstrip().startswith("```") else "Also"
            primary.a += f"\n\n{label} ({g.sources[0] if g.sources else 'other source'}):\n{g.a}"
            seen |= t
    return primary


def deduplicate(items: list[QAItem], threshold: float = 0.8) -> tuple[list[QAItem], int]:
    """Returns (unique items, number of duplicates merged away)."""
    groups = find_groups(items, threshold)
    merged = [merge_group([items[i] for i in g]) for g in groups]
    return merged, len(items) - len(merged)
