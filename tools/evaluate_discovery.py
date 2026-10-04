"""How close do automatically discovered topics get to hand-made ones?

    python tools/evaluate_discovery.py path/to/labelled_kb [--k 13]

Runs topic discovery on the questions of a labelled knowledge base WITHOUT their labels,
then compares the clusters to the hand-made topics:
  purity - share of items whose cluster's majority hand label is their own label
  NMI    - normalised mutual information between clusters and labels (0 = unrelated, 1 = identical)
and prints which hand-made topic each discovered topic mostly corresponds to.
"""
import argparse
import collections
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knowledgelake.discover import discover_topics  # noqa: E402
from knowledgelake.models import QAItem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("kb")
ap.add_argument("--k", type=int, default=None)
args = ap.parse_args()
from sklearn.metrics import normalized_mutual_info_score  # noqa: E402

items, gold = [], []
for f in sorted(glob.glob(str(Path(args.kb) / "*.json"))):
    d = json.load(open(f))
    if not (isinstance(d, dict) and "items" in d):
        continue
    for x in d["items"]:
        items.append(QAItem(q=x["q"], a=x["a"]))
        gold.append(d["topic"])
t = time.time()
cfg, found, _ = discover_topics(items, k=args.k, use_llm=False)
pairs = collections.Counter(zip(found, gold))
majority = {}
for (c, g), n in pairs.items():
    if n > majority.get(c, ("", 0))[1]:
        majority[c] = (g, n)
purity = sum(n for _, n in majority.values()) / len(gold)
print(f"{len(items)} items, {len(set(gold))} hand-made topics -> {len(cfg['topics'])} discovered "
      f"in {time.time() - t:.0f}s")
print(f"purity {purity:.1%}   NMI {normalized_mutual_info_score(gold, found):.3f}")
for c, n in collections.Counter(found).most_common():
    g, m = majority[c]
    print(f"  {n:5d}  {c:40s} -> {g} ({m / n:.0%})")
