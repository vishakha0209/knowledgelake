import csv
import tempfile
import unittest
from pathlib import Path

from knowledgelake.classify import TopicModel, classify_all_learned, load_config
from knowledgelake.dedupe import deduplicate, is_duplicate
from knowledgelake.embed import LSAEmbedder, expand
from knowledgelake.linkage import FEATURES, MODEL_PATH, PairScorer, score_pairs
from knowledgelake.models import QAItem
from knowledgelake.parse import clean_question
from knowledgelake.store import DUP_QUEUE, KnowledgeBase

CORPUS = [
    ("Explain the Medallion architecture.", "Bronze holds raw data, silver cleaned data and gold business aggregates."),
    ("What is the purpose of bronze, silver and gold layers?", "Raw data lands in bronze, silver is cleaned, gold is aggregated for BI."),
    ("What is a broadcast join?", "The small table is copied to every executor so the large table is not shuffled."),
    ("What is Change Data Capture?", "Capturing inserts, updates and deletes from a source since the last load."),
    ("How does CDC work?", "It reads the change log to pick up inserted, updated and deleted rows."),
    ("Find the second highest salary.", "```\nSELECT MAX(salary) FROM emp WHERE salary < (SELECT MAX(salary) FROM emp)\n```"),
    ("Find the second lowest salary.", "```\nSELECT MIN(salary) FROM emp WHERE salary > (SELECT MIN(salary) FROM emp)\n```"),
    ("What is a Python decorator?", "A function that wraps another function to add behaviour."),
]


class EmbeddingTests(unittest.TestCase):
    def test_abbreviations_expand(self):
        self.assertIn("change data capture", expand("How does CDC work?"))

    def test_lsa_vectors_are_normalised_and_paraphrases_are_close(self):
        emb = LSAEmbedder(dim=6).fit([q + " " + a for q, a in CORPUS])
        E = emb.encode([q for q, _ in CORPUS])
        self.assertAlmostEqual(float((E[0] ** 2).sum()), 1.0, places=5)
        self.assertGreater(float(E[3] @ E[4]), float(E[3] @ E[2]))  # CDC vs CDC  >  CDC vs broadcast join


class LinkageTests(unittest.TestCase):
    def test_feature_vector_shape(self):
        items = [QAItem(q=q, a=a, sources=["x"]) for q, a in CORPUS]
        sc = PairScorer(items, k=3, embedder=LSAEmbedder(dim=6))
        self.assertEqual(sc.matrix(sc.candidates()[:4]).shape, (4, len(FEATURES)))

    @unittest.skipUnless(MODEL_PATH.exists(), "trained pair model not present")
    def test_meaning_flip_never_scores_as_duplicate(self):
        items = [QAItem(q=q, a=a, sources=["x"]) for q, a in CORPUS]
        scored = score_pairs(items, embedder=LSAEmbedder(dim=6))
        flip = [p for p, i, j in scored if {i, j} == {5, 6}]
        self.assertTrue(not flip or flip[0] < 0.4)


class DedupeRegressionTests(unittest.TestCase):
    def test_question_made_only_of_stop_words_still_dedupes(self):
        self.assertTrue(is_duplicate("What is PySpark?", "What is PySpark?"))
        _, removed = deduplicate([QAItem(q="What is PySpark?"), QAItem(q="what is pyspark")])
        self.assertEqual(removed, 1)

    def test_common_words_count_towards_similarity(self):
        q = "Find the percentage of employees in each department."
        filler = [QAItem(q=f"Find employees in department {n} with salary above {n}") for n in range(40)]
        _, removed = deduplicate(filler + [QAItem(q=q), QAItem(q=q)])
        self.assertGreaterEqual(removed, 1)


class ParserTests(unittest.TestCase):
    def test_level_tags_become_a_field(self):
        self.assertEqual(clean_question("[Intermediate] What is schema evolution?"),
                         ("What is schema evolution?", "Intermediate"))
        self.assertEqual(clean_question("5 Q: What does Delta Lake provide?")[0], "What does Delta Lake provide?")


class LearnedClassifierTests(unittest.TestCase):
    def test_blend_sets_topic_and_confidence(self):
        cfg = load_config()
        train = ([QAItem(q=f"How do you tune a Spark shuffle partition {n}?", a="spark.sql.shuffle.partitions",
                         topic="PySpark & Spark") for n in range(10)]
                 + [QAItem(q=f"Write a SQL query with GROUP BY number {n}", a="SELECT dept, COUNT(*) FROM t GROUP BY dept",
                           topic="SQL") for n in range(10)])
        model = TopicModel().fit(train)
        items = classify_all_learned([QAItem(q="How many shuffle partitions should Spark use?")], cfg, model)
        self.assertEqual(items[0].topic, "PySpark & Spark")
        self.assertTrue(0 <= items[0].topic_confidence <= 1)


class ReviewQueueTests(unittest.TestCase):
    def test_review_decision_merges_two_items(self):
        with tempfile.TemporaryDirectory() as d:
            kb = KnowledgeBase(Path(d))
            kb.add([QAItem(q="Explain the Medallion architecture.", a="bronze silver gold", topic="Databricks & Delta Lake",
                           sources=["a"]),
                    QAItem(q="Purpose of the bronze, silver and gold layers?", a="raw, cleaned, aggregated",
                           topic="Databricks & Delta Lake", sources=["b"])], semantic=False)
            with (Path(d) / DUP_QUEUE).open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["p_duplicate", "decision", "question_a", "question_b", "topic_a", "topic_b"])
                w.writerow(["0.62", "y", "Explain the Medallion architecture.",
                            "Purpose of the bronze, silver and gold layers?", "", ""])
            res = kb.apply_reviews()
            self.assertEqual(res["duplicates_merged"], 1)
            self.assertEqual(len(kb.items), 1)
            self.assertEqual(sorted(kb.items[0].sources), ["a", "b"])
            self.assertFalse((Path(d) / DUP_QUEUE).exists())


if __name__ == "__main__":
    unittest.main()
