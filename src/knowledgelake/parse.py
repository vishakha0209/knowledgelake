"""Stage 2 - Turn raw document text into question/answer items.

Question-and-answer documents come in many shapes. The parser recognises the common ones:

* numbered items      ``12. What is a broadcast join?``  /  ``Q12 · ...``  /  ``Problem 7. ...``
* explicit markers    ``Q: ...`` / ``Question: ...``  followed by  ``A: ...`` / ``Answer: ...`` / ``Ans: ...``
* bare questions      a line ending in ``?`` with no answer (question-only lists)

Anything between one question and the next becomes the answer. Lines that look
like code (SQL / Python / PySpark) are wrapped in ``` fences so the PDF renders
them in a monospaced box.
"""
from __future__ import annotations

import re

from .models import Document, QAItem

Q_START = re.compile(
    r"""^\s*(?:
        (?:Q(?:uestion)?\s*\d*\s*[:.·\-)]\s*)        |   # Q: / Q12. / Question 3:
        (?:Problem\s+\d+\s*[.:]\s*)                  |   # Problem 7.
        (?:\d{1,3}\s*[.)]\s+)                            # 12.  or 12)
    )(?P<q>\S.*)$""",
    re.IGNORECASE | re.VERBOSE,
)
A_START = re.compile(r"^\s*(?:A|Ans|Answer|Solution)\s*[:.\-]\s*(?P<a>.*)$", re.IGNORECASE)
CODE_LINE = re.compile(
    # SQL keywords must be UPPER-CASE whole words; prose such as "Where ..." or "Deletes ..." is not code
    r"^\s*((SELECT|FROM|WHERE|GROUP BY|ORDER BY|HAVING|JOIN|LEFT JOIN|INNER JOIN|WITH|UNION|INSERT|UPDATE|DELETE|"
    r"MERGE|CREATE|ALTER|DROP|WHEN|THEN|ELSE|END|ON|AND|OR|LIMIT)\b(?!\s+[a-z]+\s+[a-z]+\s+[a-z]+\b)|"
    r"def |class \w+|import |from \w+ import|return |print\(|for \w+ in .*:$|if .*:$|elif |else:$|try:$|except|"
    r"df\.|spark\.|\w+\s*=\s*(spark|df|F\.|Window)|\.\w+\(.*\)\s*\\?$|[)\]}]\s*;?$|# |-- )"
)
LEVEL_TAG = re.compile(r"^\s*\[(foundation|beginner|basic|intermediate|advanced|expert|scenario)\]\s*", re.IGNORECASE)
LEFTOVER_MARKER = re.compile(r"^\s*(?:\d{1,3}\s+)?Q\s*\d*\s*[:.]\s*", re.IGNORECASE)   # "5 Q: What ..." from OCR
NOISE = re.compile(r"^\s*(page \d+( of \d+)?|\d+|follow .*|like .*share.*|swipe.*)\s*$", re.IGNORECASE)


def _is_question_line(line: str) -> bool:
    s = line.strip()
    return s.endswith("?") and 12 <= len(s) <= 300


def clean_question(q: str) -> tuple[str, str | None]:
    """Strip level tags ("[Intermediate] ...") into a separate field and leftover "Q:" markers."""
    level = None
    for _ in range(2):
        m = LEVEL_TAG.match(q)
        if m:
            level = m.group(1).capitalize()
            q = q[m.end():]
        q = LEFTOVER_MARKER.sub("", q, count=1)
    return q.strip(), level


def _fence_code(lines: list[str]) -> str:
    out, buf = [], []

    def flush():
        if buf:
            out.append("```\n" + "\n".join(buf).rstrip() + "\n```")
            buf.clear()

    for ln in lines:
        if CODE_LINE.match(ln) or (buf and ln.startswith((" ", "\t")) and ln.strip()):
            buf.append(ln.rstrip())
        else:
            flush()
            out.append(ln.strip())
    flush()
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_document(doc: Document) -> list[QAItem]:
    lines = [ln for ln in doc.text.replace("\f", "\n").splitlines() if not NOISE.match(ln)]
    items: list[QAItem] = []
    cur_q: list[str] | None = None
    cur_a: list[str] = []
    in_answer = False
    prev_blank = True

    def close():
        nonlocal cur_q, cur_a, in_answer
        if cur_q:
            q = re.sub(r"\s+", " ", " ".join(cur_q)).strip()
            q, level = clean_question(q)
            if len(q) >= 8:
                items.append(QAItem(q=q, a=_fence_code(cur_a), sources=[doc.source], level=level))
        cur_q, cur_a, in_answer = None, [], False

    for raw in lines:
        blank_before, prev_blank = prev_blank, not raw.strip()
        m_q = Q_START.match(raw)
        m_a = A_START.match(raw)
        if m_q and not m_a:
            close()
            cur_q = [m_q.group("q")]
            in_answer = cur_q[0].rstrip().endswith("?")
            continue
        if cur_q is None and _is_question_line(raw):
            close()
            cur_q, in_answer = [raw.strip()], True
            continue
        if cur_q is None:
            continue
        if m_a:
            in_answer = True
            if m_a.group("a").strip():
                cur_a.append(m_a.group("a"))
            continue
        if not in_answer:
            # question text that wraps onto the next line(s)
            if raw.strip() and len(cur_q) < 4 and not CODE_LINE.match(raw):
                cur_q.append(raw.strip())
                if raw.rstrip().endswith(("?", ".")):
                    in_answer = True
                continue
            in_answer = True
        if _is_question_line(raw) and (not cur_a or blank_before) and not CODE_LINE.match(raw):
            # consecutive bare questions: previous one had no answer
            close()
            cur_q, in_answer = [raw.strip()], True
            continue
        cur_a.append(raw)
    close()
    return items


def parse_documents(docs: list[Document]) -> list[QAItem]:
    out: list[QAItem] = []
    for d in docs:
        out += parse_document(d)
    return out
