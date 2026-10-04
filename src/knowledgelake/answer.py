"""Stage 5 - Fill in answers for question-only items with an LLM.

Uses the Anthropic Messages API over plain HTTPS (no SDK dependency). Set
ANTHROPIC_API_KEY to enable; without a key the step is skipped and the items
are kept with an empty answer so they show up as "to answer" in the PDF.
Every generated answer is flagged `answered_by_llm=True` and rendered with a
marker, so AI-written content is always distinguishable from the source notes.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request

from .models import QAItem

log = logging.getLogger(__name__)
API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = os.environ.get("KB_MODEL", "claude-sonnet-4-5")

PROMPT = """You are a senior data engineer writing entries for a technical knowledge base.
Topic: {topic}
Question: {q}

Write a concise, clear answer (80-200 words). Lead with the direct answer,
then key points or trade-offs. If a query or code is the natural answer, include it
in a ``` code block. Do not add a heading or repeat the question."""


def _call(prompt: str, api_key: str, model: str, retries: int = 3) -> str:
    body = json.dumps({"model": model, "max_tokens": 700,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            return "".join(b.get("text", "") for b in data.get("content", [])).strip()
        except Exception as exc:  # network / rate limit: back off and retry
            wait = 2 ** attempt * 5
            log.warning("LLM call failed (%s), retrying in %ss", exc, wait)
            time.sleep(wait)
    return ""


def answer_missing(items: list[QAItem], model: str = DEFAULT_MODEL) -> int:
    key = os.environ.get("ANTHROPIC_API_KEY")
    todo = [i for i in items if not i.has_answer]
    if not todo:
        return 0
    if not key:
        log.info("%d question-only items left unanswered (set ANTHROPIC_API_KEY to fill them)", len(todo))
        return 0
    done = 0
    for item in todo:
        text = _call(PROMPT.format(topic=item.topic, q=item.q), key, model)
        if text:
            item.a, item.answered_by_llm = text, True
            done += 1
    return done
