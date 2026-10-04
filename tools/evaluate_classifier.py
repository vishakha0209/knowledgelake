"""Measure topic-classification accuracy against a hand-labelled knowledge base.

    python tools/evaluate_classifier.py path/to/labelled_kb [--config topics.json] [--learned]

Without --learned: keyword rules + document prior (no training involved).
With --learned: rules blended with the TF-IDF logistic-regression TopicModel, evaluated
with 5-fold GroupKFold *by source file*, so no file ever contributes to both training
and testing (a random split would leak each file's style and overstate accuracy).
Also prints how accuracy changes when the least confident items are flagged for review.
"""
import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knowledgelake.classify import TopicModel, classify_all, classify_all_learned, load_config  # noqa: E402
from knowledgelake.models import QAItem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("kb")
ap.add_argument("--config")
ap.add_argument("--learned", action="store_true")
args = ap.parse_args()
cfg = load_config(args.config)

rows = []
for f in sorted(glob.glob(str(Path(args.kb) / "*.json"))):
    d = json.load(open(f))
    if not (isinstance(d, dict) and "items" in d):
        continue
    for x in d["items"]:
        rows.append((x["q"], x["a"], (x.get("sources") or [""])[0], d["topic"]))
gold = np.array([r[3] for r in rows])


def fresh():
    return [QAItem(q=q, a=a, sources=[s]) for q, a, s, _ in rows]


items = classify_all(fresh(), cfg)
pred = np.array([i.topic for i in items])
print(f"keyword rules + document prior: {(pred == gold).mean():.1%}  ({(pred == gold).sum()}/{len(gold)})")

if args.learned:
    from sklearn.model_selection import GroupKFold
    items = fresh()
    conf = np.zeros(len(items))
    for tr, te in GroupKFold(n_splits=5).split(rows, gold, [r[2] for r in rows]):
        train = [QAItem(q=rows[k][0], a=rows[k][1], topic=rows[k][3]) for k in tr]
        test = [items[k] for k in te]
        classify_all_learned(test, cfg, TopicModel().fit(train))
    pred = np.array([i.topic for i in items])
    conf = np.array([i.topic_confidence for i in items])
    ok = pred == gold
    print(f"learned model + rules + document prior (leave-source-out CV): {ok.mean():.1%}")
    for q in (0.1, 0.2, 0.3):
        th = np.quantile(conf, q)
        flag = conf < th
        print(f"  flag least-confident {q:.0%} (confidence < {th:.3f}): accuracy on the rest {ok[~flag].mean():.1%}, "
              f"flagged items contain {(~ok[flag]).sum()}/{(~ok).sum()} of the errors")

print("most common confusions (true -> predicted):")
for (g, p), n in collections.Counter((g, p) for g, p in zip(gold, pred) if g != p).most_common(6):
    print(f"  {n:4d}  {g}  ->  {p}")
