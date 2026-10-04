"""Stage 6 - Persist the knowledge base as one JSON file per topic and append incrementally.

Appending is idempotent: re-ingesting the same notes adds nothing, because every
new question is checked against the questions already stored before it is added:
lexical rules (stage 4) plus, when enabled, the semantic pair model (stage 4b).

Uncertain decisions are not guessed silently. They go to two CSV review queues in the
KB folder - review_duplicates.csv and review_topics.csv - which `kb review` applies.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import re
from pathlib import Path

from .dedupe import deduplicate, find_groups, merge_group
from .models import QAItem

log = logging.getLogger(__name__)
DUP_QUEUE, TOPIC_QUEUE = "review_duplicates.csv", "review_topics.csv"
CONFIG_FILE = "kb_config.json"   # the topics this KB files into (a preset or discovered); edit to rename topics
TOPIC_REVIEW_BELOW = 0.18   # ~ the least confident 20% on my notes (tools/evaluate_classifier.py)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def slug(topic: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", topic).strip("_")


class KnowledgeBase:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.topics: dict[str, dict] = {}
        for f in sorted(self.root.glob("*.json")):
            d = json.loads(f.read_text())
            if not (isinstance(d, dict) and "topic" in d and "items" in d):
                continue  # not a knowledge-base file
            d["items"] = [QAItem.from_dict(x) for x in d["items"]]
            self.topics[d["topic"]] = d

    @property
    def config(self) -> dict | None:
        f = self.root / CONFIG_FILE
        return json.loads(f.read_text()) if f.exists() else None

    def save_config(self, cfg: dict) -> None:
        (self.root / CONFIG_FILE).write_text(json.dumps(cfg, indent=1, ensure_ascii=False))

    @property
    def items(self) -> list[QAItem]:
        return [i for t in self.topics.values() for i in t["items"]]

    def _file(self, item: QAItem) -> None:
        t = self.topics.setdefault(item.topic, {"topic": item.topic, "section_order": [], "items": []})
        item.id = max((x.id for x in t["items"]), default=0) + 1
        t["items"].append(item)
        if item.section not in t["section_order"]:
            t["section_order"].append(item.section)

    def _remove(self, item: QAItem) -> None:
        t = self.topics[item.topic]
        t["items"] = [x for x in t["items"] if x is not item]
        if not t["items"]:
            del self.topics[item.topic]
            (self.root / f"{slug(item.topic)}.json").unlink(missing_ok=True)

    def add(self, new_items: list[QAItem], section_order: dict[str, list[str]] | None = None,
            semantic: bool = True, review_min: float | None = None) -> dict:
        """Merge new items into the KB. Returns counts for reporting."""
        today = dt.date.today().isoformat()
        new_items, internal_dups = deduplicate(new_items)
        existing = self.items
        combined = self.last_combined = existing + new_items
        n_existing = len(existing)

        auto_pairs, queue = [], []
        if semantic and new_items:
            from .linkage import score_pairs, split_decisions
            scored = score_pairs(combined, focus=set(range(n_existing, len(combined))))
            auto_pairs, queue = split_decisions(scored, review=review_min) if review_min is not None \
                else split_decisions(scored)

        groups = find_groups(combined, extra_pairs=auto_pairs)
        lexical_only = {frozenset(g) for g in find_groups(combined) if len(g) > 1}
        added, merged_into_existing, semantic_merges = 0, 0, 0
        for g in groups:
            olds = [combined[i] for i in g if i < n_existing]
            news = [combined[i] for i in g if i >= n_existing]
            if not news:
                continue
            if len(g) > 1 and frozenset(g) not in lexical_only:
                semantic_merges += len(news) if olds else len(news) - 1
            if olds:  # already known: keep old record, pick up new sources / extra info
                merge_group(olds[:1] + news)
                merged_into_existing += len(news)
                continue
            item = merge_group(news)
            item.added = today
            self._file(item)
            added += 1
        if section_order:
            for topic, order in section_order.items():
                if topic in self.topics:
                    rest = [s for s in self.topics[topic]["section_order"] if s not in order]
                    self.topics[topic]["section_order"] = list(order) + rest

        pending_dups = 0
        if queue:
            from .linkage import write_queue
            live = {id(x) for x in self.items}
            queue = [(p, i, j) for p, i, j in queue if id(combined[i]) in live and id(combined[j]) in live]
            pending_dups = write_queue(self.root / DUP_QUEUE, queue, combined)
        low = [i for i in new_items if i.topic_confidence is not None and i.topic_confidence < TOPIC_REVIEW_BELOW
               and any(i is x for x in self.items)]
        pending_topics = self._write_topic_queue(low) if low else 0
        return {"added": added, "duplicates_within_batch": internal_dups,
                "already_in_kb": merged_into_existing, "semantic_merges": semantic_merges,
                "review_duplicates_pending": pending_dups, "review_topics_pending": pending_topics}

    def _write_topic_queue(self, items: list[QAItem]) -> int:
        path = self.root / TOPIC_QUEUE
        rows = _read_csv(path) if path.exists() else []
        seen = {r["question"] for r in rows}
        for i in items:
            if i.q not in seen:
                rows.append({"confidence": f"{i.topic_confidence:.2f}", "question": i.q, "topic": i.topic,
                             "correct_topic": ""})
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["confidence", "question", "topic", "correct_topic"])
            w.writeheader()
            w.writerows(rows)
        return sum(1 for r in rows if not r["correct_topic"])

    def rename_topic(self, old: str, new: str) -> int:
        """Rename a topic, or merge it into another one when `new` already exists. Returns items moved."""
        if old not in self.topics:
            raise KeyError(f"no topic named {old!r}; topics are: {', '.join(sorted(self.topics))}")
        moved = list(self.topics[old]["items"])
        for item in moved:
            self._remove(item)
            item.topic = new
            self._file(item)
        cfg = self.config
        if cfg and old in cfg.get("topics", {}):
            spec = cfg["topics"].pop(old)
            if new in cfg["topics"]:   # merging: keep both keyword sets
                for k, v in spec.get("keywords", {}).items():
                    cfg["topics"][new].setdefault("keywords", {})[k] = max(v, cfg["topics"][new]["keywords"].get(k, 0))
                cfg["topics"][new].setdefault("sections", {}).update(spec.get("sections", {}))
            else:
                cfg["topics"][new] = spec
            order = cfg.get("section_order", {})
            if old in order:
                order[new] = order.get(new, []) + [s for s in order.pop(old) if s not in order.get(new, [])]
            self.save_config(cfg)
        return len(moved)

    def apply_reviews(self, cfg: dict | None = None) -> dict:
        """Apply decisions filled into the two review CSVs, then remove the decided rows."""
        by_q = {}
        for i in self.items:
            by_q.setdefault(i.q, i)
        merged = moved = 0
        path = self.root / DUP_QUEUE
        if path.exists():
            rows = _read_csv(path)
            keep = []
            for r in rows:
                d = r["decision"].strip().lower()
                a, b = by_q.get(r["question_a"]), by_q.get(r["question_b"])
                if d in ("y", "yes") and a and b and a is not b:
                    self._remove(a)
                    self._remove(b)
                    primary = merge_group([a, b])
                    self._file(primary)
                    by_q[r["question_a"]] = by_q[r["question_b"]] = primary
                    merged += 1
                elif not d:
                    keep.append(r)
            self._rewrite(path, rows[0].keys() if rows else [], keep)
        path = self.root / TOPIC_QUEUE
        if path.exists():
            rows = _read_csv(path)
            keep = []
            for r in rows:
                new = r["correct_topic"].strip()
                item = by_q.get(r["question"])
                if new and item:
                    if new != item.topic:
                        self._remove(item)
                        item.topic = new
                        if cfg:
                            from .classify import _section
                            _section(item, cfg)
                        self._file(item)
                        moved += 1
                    item.topic_confidence = None
                elif not new:
                    keep.append(r)
            self._rewrite(path, ["confidence", "question", "topic", "correct_topic"], keep)
        return {"duplicates_merged": merged, "topics_changed": moved}

    @staticmethod
    def _rewrite(path: Path, fields, rows) -> None:
        if not rows:
            path.unlink(missing_ok=True)
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fields))
            w.writeheader()
            w.writerows(rows)

    def save(self) -> None:
        for topic, d in self.topics.items():
            out = {"topic": topic, "section_order": d["section_order"],
                   "items": [i.to_dict() for i in d["items"]]}
            (self.root / f"{slug(topic)}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    def stats(self) -> dict:
        per_topic = {t: len(d["items"]) for t, d in self.topics.items()}
        by_llm = sum(1 for i in self.items if i.answered_by_llm)
        unanswered = sum(1 for i in self.items if not i.has_answer)
        return {"total": sum(per_topic.values()), "per_topic": per_topic,
                "answered_by_llm": by_llm, "unanswered": unanswered}
