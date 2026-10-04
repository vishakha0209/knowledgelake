"""Core data structures shared by every stage of the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Document:
    """Raw text pulled out of one input file."""
    source: str            # file name the text came from
    text: str
    method: str            # "text-layer", "ocr-pdf", "ocr-image", "plain-text"
    pages: int = 1


@dataclass
class QAItem:
    """One question/answer item (the answer may still be missing)."""
    q: str
    a: str = ""
    topic: str = "Uncategorised"
    section: str = "General"
    level: Optional[str] = None
    sources: list[str] = field(default_factory=list)
    answered_by_llm: bool = False
    added: str = ""
    id: int = 0
    topic_confidence: Optional[float] = None   # set by the learned classifier; low = worth a human look

    @property
    def has_answer(self) -> bool:
        return bool(self.a.strip())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QAItem":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        # backwards compatibility with the first hand-built knowledge base
        if "answered_by_claude" in d and "answered_by_llm" not in d:
            known["answered_by_llm"] = d["answered_by_claude"]
        return cls(**known)
