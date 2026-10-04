"""Command-line entry point.

    kb ingest notes.zip more_notes/ --kb kb_store      # extract -> parse -> classify -> dedupe -> answer -> store
    kb build --kb kb_store --out pdfs                  # render one PDF per topic
    kb stats --kb kb_store
    kb topics --kb kb_store                            # topics with counts (discovered or preset)
    kb rename --kb kb_store "VPN & Account" "Remote Access"   # rename, or merge into an existing topic
    kb discover my_docs/ --out topics.json             # preview / hand-edit topics before ingesting
    kb review --kb kb_store                            # apply decisions from the review CSVs
    kb chat --kb kb_store                              # "Ask my notes" web app on localhost
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .answer import answer_missing
from .classify import TopicModel, classify_all, classify_all_learned, load_config
from .extract import extract_inputs
from .parse import parse_documents
from .render import build_all
from .store import KnowledgeBase


def resolve_topics(kb: KnowledgeBase, choice: str | None) -> tuple[dict | None, str]:
    """Which topics to file into. Returns (config or None meaning 'discover from this batch', source).

    1. an explicit preset name or JSON path always wins;
    2. otherwise the KB's own saved config (so every batch lands in the same topics);
    3. a KB built before configs were saved uses the data-engineering preset (backwards compatible);
    4. a new KB with "auto" (the default) discovers topics from the first batch.
    """
    if choice and choice != "auto":
        return load_config(choice), f"preset/file: {choice}"
    if kb.config:
        return kb.config, "saved in knowledge base"
    if kb.items:
        return load_config(None), "data-engineering preset (existing knowledge base)"
    return None, "discovered from this batch"


def cmd_ingest(args) -> int:
    t0 = time.time()
    docs = extract_inputs([Path(p) for p in args.inputs])
    methods = {}
    for d in docs:
        methods[d.method] = methods.get(d.method, 0) + 1
    items = parse_documents(docs)
    kb = KnowledgeBase(Path(args.kb))
    cfg, source = resolve_topics(kb, args.topics or args.config)
    model = None
    if cfg is None:
        from .discover import discover_topics
        cfg, topic_of, section_of = discover_topics(items, k=args.k)
        for item, t, sec in zip(items, topic_of, section_of):
            item.topic, item.section = t, sec
        classifier = "discovered"
    else:
        model = None if args.no_learned else TopicModel.from_kb(kb.items)
        if model:
            classify_all_learned(items, cfg, model)
        else:
            classify_all(items, cfg)
        classifier = "learned+rules" if model else "rules"
    if kb.config != cfg:
        kb.save_config(cfg)
    report = kb.add(items, cfg.get("section_order"), semantic=not args.no_semantic, review_min=args.review_min)
    report["topics"] = {"source": source, "classifier": classifier, "count": len(cfg["topics"])}
    answered = 0 if args.no_llm else answer_missing([i for i in kb.items if not i.has_answer])
    kb.save()
    summary = {"files": len(docs), "extraction": methods, "questions_found": len(items),
               **report, "llm_answers": answered, "kb": kb.stats(), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(summary, indent=2))
    if args.build:
        for fn, n in build_all(args.kb, args.out):
            print(f"{n:5d}  {fn}")
    return 0


def cmd_review(args) -> int:
    kb = KnowledgeBase(Path(args.kb))
    res = kb.apply_reviews(kb.config or load_config(args.config))
    kb.save()
    print(json.dumps(res, indent=2))
    if args.build:
        for fn, n in build_all(args.kb, args.out):
            print(f"{n:5d}  {fn}")
    return 0


def cmd_topics(args) -> int:
    from collections import Counter
    from .discover import summarise
    kb = KnowledgeBase(Path(args.kb))
    cfg = kb.config or load_config(None)
    print(summarise(cfg, Counter(i.topic for i in kb.items)))
    print('\nRename or merge:  kb rename --kb KB "Old name" "New name"   (merges if the new name exists)')
    return 0


def cmd_rename(args) -> int:
    kb = KnowledgeBase(Path(args.kb))
    n = kb.rename_topic(args.old, args.new)
    kb.save()
    print(f"moved {n} items: {args.old!r} -> {args.new!r}")
    return 0


def cmd_discover(args) -> int:
    from collections import Counter
    from .discover import discover_topics, summarise
    items = parse_documents(extract_inputs([Path(p) for p in args.inputs]))
    cfg, topic_of, _ = discover_topics(items, k=args.k)
    print(summarise(cfg, Counter(topic_of)))
    if args.out:
        Path(args.out).write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
        print(f"\nsaved {args.out}: edit names if you like, then `kb ingest ... --topics {args.out}`")
    return 0


def cmd_chat(args) -> int:
    from .chat import serve
    serve(Path(args.kb), host=args.host, port=args.port)
    return 0


def cmd_build(args) -> int:
    for fn, n in build_all(args.kb, args.out):
        print(f"{n:5d}  {fn}")
    return 0


def cmd_stats(args) -> int:
    print(json.dumps(KnowledgeBase(Path(args.kb)).stats(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kb", description="Messy documents -> a de-duplicated, topic-wise, searchable knowledge base")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest", help="add notes (files, folders or .zip) to the knowledge base")
    i.add_argument("inputs", nargs="+")
    i.add_argument("--kb", default="kb_store")
    i.add_argument("--topics", default=None,
                   help="auto (default: discover topics from the first batch), a preset name "
                        "(data-engineering) or a topics JSON file")
    i.add_argument("--config", default=None, help=argparse.SUPPRESS)  # old name for --topics
    i.add_argument("--k", type=int, default=None, help="number of topics to discover (default: chosen automatically)")
    i.add_argument("--no-llm", action="store_true", help="skip answering question-only items")
    i.add_argument("--build", action="store_true", help="render PDFs after ingesting")
    i.add_argument("--out", default="pdfs")
    i.add_argument("--no-semantic", action="store_true", help="lexical de-duplication only")
    i.add_argument("--review-min", type=float, default=None,
                   help="lowest duplicate probability sent to the review queue (default 0.4; "
                        "try 0.05 for small document sets, whose short questions score lower)")
    i.add_argument("--no-learned", action="store_true", help="keyword rules only for topics")
    i.set_defaults(func=cmd_ingest)

    r = sub.add_parser("review", help="apply decisions filled into review_duplicates.csv / review_topics.csv")
    r.add_argument("--kb", default="kb_store")
    r.add_argument("--config", default=None)
    r.add_argument("--build", action="store_true")
    r.add_argument("--out", default="pdfs")
    r.set_defaults(func=cmd_review)

    t = sub.add_parser("topics", help="list the knowledge base's topics with counts and keywords")
    t.add_argument("--kb", default="kb_store")
    t.set_defaults(func=cmd_topics)

    rn = sub.add_parser("rename", help="rename a topic, or merge it into an existing one")
    rn.add_argument("old")
    rn.add_argument("new")
    rn.add_argument("--kb", default="kb_store")
    rn.set_defaults(func=cmd_rename)

    d = sub.add_parser("discover", help="preview the topics that would be discovered from some documents")
    d.add_argument("inputs", nargs="+")
    d.add_argument("--k", type=int, default=None)
    d.add_argument("--out", default=None, help="save the discovered topics config to this JSON file")
    d.set_defaults(func=cmd_discover)

    c = sub.add_parser("chat", help='"Ask my notes": retrieval over the KB, answers from Claude when a key is set')
    c.add_argument("--kb", default="kb_store")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=8501)
    c.set_defaults(func=cmd_chat)

    b = sub.add_parser("build", help="render PDFs from the knowledge base")
    b.add_argument("--kb", default="kb_store")
    b.add_argument("--out", default="pdfs")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("stats", help="print knowledge-base statistics")
    s.add_argument("--kb", default="kb_store")
    s.set_defaults(func=cmd_stats)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
