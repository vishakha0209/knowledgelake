import tempfile
import unittest
from pathlib import Path
from unittest import mock

from knowledgelake.embed import LSAEmbedder
from knowledgelake.models import QAItem
from knowledgelake.rag import Retriever, answer
from knowledgelake.store import KnowledgeBase

NOTES = [
    QAItem(q="What is a broadcast join?", a="Small table copied to every executor; avoids a shuffle.", topic="PySpark & Spark"),
    QAItem(q="What is Change Data Capture?", a="Capturing inserts, updates and deletes since the last load.", topic="Data Engineering Concepts & Architecture"),
    QAItem(q="ROW_NUMBER vs RANK vs DENSE_RANK?", a="RANK leaves gaps after ties, DENSE_RANK does not.", topic="SQL"),
    QAItem(q="Explain the Medallion architecture.", a="Bronze raw, silver cleaned, gold aggregated.", topic="Databricks & Delta Lake"),
]


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.r = Retriever(NOTES, embedder=LSAEmbedder(dim=3))

    def test_abbreviation_query_finds_note(self):
        self.assertEqual(self.r.search("how does CDC work", k=1)[0].item.q, "What is Change Data Capture?")

    def test_topic_filter(self):
        hits = self.r.search("join rank medallion", k=5, topic="SQL")
        self.assertTrue(hits and all(h.item.topic == "SQL" for h in hits))

    def test_answer_without_key_returns_notes_only(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(answer("q", self.r.search("rank")), {"answer": None, "generated": False})

    def test_answer_prompt_contains_numbered_notes(self):
        with mock.patch("knowledgelake.answer._call", return_value="RANK leaves gaps [1].") as call:
            out = answer("rank vs dense_rank", self.r.search("dense rank", k=2), api_key="test")
        self.assertTrue(out["generated"])
        self.assertIn("[1] Q: ROW_NUMBER vs RANK vs DENSE_RANK?", call.call_args[0][0])


class ChatAppTests(unittest.TestCase):
    def test_ask_endpoint(self):
        try:
            import flask  # noqa: F401
        except ImportError:
            self.skipTest("flask not installed")
        from knowledgelake.chat import create_app
        with tempfile.TemporaryDirectory() as d:
            kb = KnowledgeBase(Path(d))
            kb.add([QAItem(q=n.q, a=n.a, topic=n.topic) for n in NOTES], semantic=False)
            kb.save()
            c = create_app(Path(d), retriever=Retriever(kb.items, embedder=LSAEmbedder(dim=3))).test_client()
            r = c.post("/api/ask", json={"question": "what is a broadcast join"}).get_json()
            self.assertEqual(r["hits"][0]["q"], "What is a broadcast join?")
            self.assertEqual(c.post("/api/ask", json={}).status_code, 400)
