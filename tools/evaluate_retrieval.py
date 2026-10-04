"""Evaluate "Ask my notes" retrieval with real paraphrases.

    python tools/evaluate_retrieval.py labelled_pairs.json path/to/kb

Every duplicate group in the labelled set was merged into one KB item. The *other*
wordings of that question (the ones that didn't survive the merge) are natural
paraphrased queries whose correct answer is known: the KB item they were merged into.
Reports Recall@1/@3/@5 and MRR for keyword-only, semantic-only and hybrid (RRF) retrieval.
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knowledgelake.rag import Retriever  # noqa: E402
from knowledgelake.store import KnowledgeBase  # noqa: E402

labelled = json.load(open(sys.argv[1]))["items"]
kb_items = KnowledgeBase(Path(sys.argv[2])).items
kb_index = {i.q: n for n, i in enumerate(kb_items)}

clusters = collections.defaultdict(list)
for x in labelled:
    clusters[x["cluster"]].append(x["q"])
queries = []
for qs in clusters.values():
    targets = [kb_index[q] for q in qs if q in kb_index]
    if len(targets) != 1:
        continue
    t = targets[0]
    queries += [(q, t) for q in qs if q != kb_items[t].q and q.strip().lower() != kb_items[t].q.strip().lower()]
print(f"{len(kb_items)} KB items, {len(queries)} paraphrased queries with a known answer")

for mode in ("keyword", "semantic", "hybrid"):
    r = Retriever(kb_items, mode=mode)
    hits_at = collections.Counter()
    mrr = 0.0
    for q, t in queries:
        ranked = [kb_index[h.item.q] for h in r.search(q, k=10)]
        if t in ranked:
            pos = ranked.index(t) + 1
            mrr += 1 / pos
            for k in (1, 3, 5):
                hits_at[k] += pos <= k
    n = len(queries)
    print(f"{mode:9s} Recall@1 {hits_at[1] / n:.1%}  Recall@3 {hits_at[3] / n:.1%}  "
          f"Recall@5 {hits_at[5] / n:.1%}  MRR {mrr / n:.3f}")
