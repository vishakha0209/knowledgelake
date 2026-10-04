"""Train and evaluate the duplicate-pair model from a hand-labelled set.

    python tools/train_pair_model.py labelled_pairs.json [--save]

labelled_pairs.json = {"items": [{"q": ..., "a": ..., "source": ..., "cluster": <id>}, ...]}
Items with the same cluster id are duplicates of each other (the result of a manual
review). My own labelled set comes from my notes and is not in the repo; the trained
model that ships in src/knowledgelake/data/pair_model.json contains only numeric feature weights.

Prints cross-validated results (folds split by cluster, so a duplicate group is
never in both train and test):
  * end-to-end pairwise precision / recall / F1 after union-find clustering
  * reviewer effort: pairs to check to find 50/70/80% of the duplicates,
    ranking by the model vs ranking by token Jaccard (the lexical baseline)
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knowledgelake.dedupe import find_groups  # noqa: E402
from knowledgelake.linkage import FEATURES, MODEL_PATH, PairModel, PairScorer, expand_features  # noqa: E402
from knowledgelake.models import QAItem  # noqa: E402


def cluster_pairs(labels):
    d = collections.defaultdict(list)
    for k, c in enumerate(labels):
        d[c].append(k)
    return {(a, b) for v in d.values() for x, a in enumerate(v) for b in v[x + 1:]}


def transitive(pairs, n):
    par = list(range(n))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in pairs:
        par[f(a)] = f(b)
    return cluster_pairs([f(k) for k in range(n)])


def prf(pred, gold):
    tp = len(pred & gold)
    p, r = tp / max(len(pred), 1), tp / max(len(gold), 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labelled")
    ap.add_argument("--save", action="store_true", help=f"fit on everything and save to {MODEL_PATH}")
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fit(Xtr, ytr) -> PairModel:
        Z = expand_features(Xtr)
        sc_ = StandardScaler().fit(Z)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc_.transform(Z), ytr)
        return PairModel(sc_.mean_, sc_.scale_, lr.coef_[0], lr.intercept_[0])

    data = json.load(open(args.labelled))["items"]
    items = [QAItem(q=x["q"], a=x.get("a", ""), sources=[x.get("source", "")]) for x in data]
    labels = [x["cluster"] for x in data]
    gold = cluster_pairs(labels)
    n = len(items)
    print(f"{n} items, {len(set(labels))} gold clusters, {len(gold)} duplicate pairs")

    lex = set()
    for g in find_groups(items):
        lex |= {(min(g[0], x), max(g[0], x)) for x in g[1:]}
    p, r, f1 = prf(transitive(lex, n), gold)
    print(f"lexical baseline (dedupe.py):     P={p:.3f} R={r:.3f} F1={f1:.3f}")

    t = time.time()
    sc = PairScorer(items)
    cand = sc.candidates(extra=lex)
    X, y = sc.matrix(cand), np.array([c in gold for c in cand])
    print(f"{len(cand)} candidate pairs from k-NN blocking; they contain {y.sum()}/{len(gold)} "
          f"gold pairs ({y.sum() / len(gold):.0%}); features in {time.time() - t:.0f}s")

    fold = np.array([hash(labels[i]) % 5 for i, _ in cand])
    prob = np.zeros(len(cand))
    for k in range(5):
        m = fit(X[fold != k], y[fold != k])
        prob[fold == k] = m.predict_proba(X[fold == k])[:, 1]

    print("\ncross-validated, after clustering:")
    for th in (0.4, 0.6, 0.8):
        sel = {c for c, pi in zip(cand, prob) if pi >= th}
        p, r, f1 = prf(transitive(sel | lex, n), gold)
        print(f"  model p>={th} + lexical:        P={p:.3f} R={r:.3f} F1={f1:.3f}")

    # reviewer effort: rank all lexically-related pairs by Jaccard vs candidates by model probability
    toks, crit = sc.toks, sc.crit
    inv = collections.defaultdict(set)
    for k, s in enumerate(toks):
        for tkn in s:
            inv[tkn].add(k)
    lexpairs = {(i, j) for i, s in enumerate(toks) for tkn in s if len(inv[tkn]) <= 300 for j in inv[tkn] if j > i}
    jac = lambda p_: len(toks[p_[0]] & toks[p_[1]]) / max(len(toks[p_[0]] | toks[p_[1]]), 1)  # noqa: E731
    lexrank = sorted(lexpairs, key=lambda p_: (crit[p_[0]] == crit[p_[1]], jac(p_)), reverse=True)
    modrank = [c for _, c in sorted(zip(-prob, cand))]

    def need(rank, target):
        found = 0
        for k, p_ in enumerate(rank, 1):
            found += p_ in gold
            if found >= target * len(gold):
                return k
        return None
    print("\nreviewer effort - pairs to check to find X% of the duplicates:")
    for target in (0.5, 0.7, 0.8):
        print(f"  {target:.0%}:  lexical ranking {need(lexrank, target)}   model ranking {need(modrank, target)}")

    if args.save:
        m = fit(X, y)
        m.meta = {"trained_on": f"{n} labelled items, {len(gold)} duplicate pairs", "model": "logistic regression"}
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_text(json.dumps(m.to_json(), indent=1))
        print(f"\nsaved {MODEL_PATH}  (features: {', '.join(FEATURES)})")


if __name__ == "__main__":
    main()
