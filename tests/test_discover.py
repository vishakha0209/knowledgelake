import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from knowledgelake.classify import load_config, preset_names
from knowledgelake.cli import main, resolve_topics
from knowledgelake.discover import _name, discover_topics, distinctive_terms
from knowledgelake.models import QAItem
from knowledgelake.store import CONFIG_FILE, KnowledgeBase

GROUPS = {
    "vpn": [("How do I connect to the VPN?", "Open the VPN client and sign in."),
            ("The VPN keeps disconnecting.", "Update the VPN client and use wired network."),
            ("Which VPN region should I pick?", "Pick the nearest VPN region."),
            ("VPN is slow from home", "Choose a closer VPN region and restart the VPN client."),
            ("Can I use the VPN on my phone?", "Yes, install the mobile VPN client.")],
    "leave": [("How many days of annual leave do I get?", "24 days of annual leave per year."),
              ("How do I apply for leave?", "Request leave in the HR portal; your manager approves the leave."),
              ("What is the sick leave policy?", "Up to 10 paid sick leave days."),
              ("Does unused leave carry over?", "Up to 5 days of leave carry over."),
              ("What is parental leave?", "26 weeks of paid parental leave.")],
    "expenses": [("How do I claim travel expenses?", "Submit the expense claim with receipts in the expense tool."),
                 ("What is the meal allowance?", "Meal expenses up to 40 per day, claim in the expense tool."),
                 ("How fast are expense claims paid?", "Expense claims are paid with the next salary."),
                 ("Can I expense a taxi?", "Taxi expenses are fine with a receipt in the expense tool."),
                 ("Who approves my expense claim?", "Your manager approves each expense claim.")],
}


def demo_items():
    return [QAItem(q=q, a=a, sources=[g]) for g, qa in GROUPS.items() for q, a in qa]


class DiscoveryTests(unittest.TestCase):
    def test_finds_the_three_groups(self):
        items = demo_items()
        cfg, topics, _ = discover_topics(items, k=3, use_llm=False)
        self.assertEqual(len(cfg["topics"]), 3)
        for g in GROUPS:  # every group lands in exactly one discovered topic
            self.assertEqual(len({t for t, i in zip(topics, items) if i.sources[0] == g}), 1, g)

    def test_names_come_from_distinctive_terms(self):
        cfg, _, _ = discover_topics(demo_items(), k=3, use_llm=False)
        names = " ".join(cfg["topics"]).lower()
        for word in ("vpn", "leave", "expense"):
            self.assertIn(word, names)

    def test_name_skips_code_like_and_repeated_terms(self):
        name = _name([("password", 3), ("reset password", 2), ("e.customer_id", 2), ("vpn", 1)], set())
        self.assertEqual(name, "Password & VPN")

    def test_tiny_input_falls_back_to_general(self):
        cfg, topics, _ = discover_topics([QAItem(q="What is X?")], use_llm=False)
        self.assertEqual(topics, ["General"])

    def test_distinctive_terms_ignore_question_words(self):
        terms = distinctive_terms(["what is the vpn", "explain the leave policy"], [0, 1])
        self.assertNotIn("what", [t for t, _ in terms[0]])

    def test_llm_names_used_when_key_present(self):
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}), \
             mock.patch("knowledgelake.answer._call", return_value='{"0": "Remote Access", "1": "Time Off", "2": "Expenses"}'):
            cfg, _, _ = discover_topics(demo_items(), k=3)
        self.assertEqual(set(cfg["topics"]), {"Remote Access", "Time Off", "Expenses"})


class RenameTests(unittest.TestCase):
    def test_rename_and_merge(self):
        with tempfile.TemporaryDirectory() as d:
            kb = KnowledgeBase(Path(d))
            kb.save_config({"topics": {"A": {"keywords": {"vpn": 2}}, "B": {"keywords": {"leave": 2}}}})
            kb.add([QAItem(q="How do I connect to the VPN?", topic="A"), QAItem(q="How much leave do I get?", topic="B")],
                   semantic=False)
            kb.save()
            self.assertEqual(kb.rename_topic("A", "Remote Access"), 1)
            self.assertEqual(kb.rename_topic("B", "Remote Access"), 1)       # merge
            kb.save()
            kb = KnowledgeBase(Path(d))
            self.assertEqual(set(kb.topics), {"Remote Access"})
            self.assertEqual(set(kb.config["topics"]["Remote Access"]["keywords"]), {"vpn", "leave"})
            self.assertEqual(sorted(p.name for p in Path(d).glob("*.json")), ["Remote_Access.json", CONFIG_FILE])


class TopicResolutionTests(unittest.TestCase):
    def test_presets(self):
        self.assertIn("data-engineering", preset_names())
        self.assertIn("SQL", load_config("data-engineering")["topics"])

    def test_resolution_order(self):
        with tempfile.TemporaryDirectory() as d:
            kb = KnowledgeBase(Path(d))
            self.assertIsNone(resolve_topics(kb, None)[0])            # new KB: discover
            self.assertIn("SQL", resolve_topics(kb, "data-engineering")[0]["topics"])
            kb.save_config({"topics": {"Mine": {"keywords": {}}}})
            self.assertEqual(list(resolve_topics(kb, None)[0]["topics"]), ["Mine"])   # saved config wins

    def test_ingest_auto_saves_config_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "docs.txt"
            src.write_text("\n\n".join(f"Q: {q}\nA: {a}" for qa in GROUPS.values() for q, a in qa))
            kb_dir = Path(d) / "kb"
            with mock.patch("builtins.print"):
                main(["ingest", str(src), "--kb", str(kb_dir), "--no-llm", "--no-semantic", "--k", "3"])
            saved = json.loads((kb_dir / CONFIG_FILE).read_text())
            self.assertEqual(len(saved["topics"]), 3)
            more = Path(d) / "more.txt"
            more.write_text("Q: How do I reset the VPN client?\nA: Reinstall the VPN client.")
            with mock.patch("builtins.print"):
                main(["ingest", str(more), "--kb", str(kb_dir), "--no-llm", "--no-semantic"])
            kb = KnowledgeBase(kb_dir)
            new = [i for i in kb.items if "reset the VPN" in i.q][0]
            vpn_topic = [i.topic for i in kb.items if i.q == "How do I connect to the VPN?"][0]
            self.assertEqual(new.topic, vpn_topic)      # later batch filed into the discovered topic
            from knowledgelake.render import build_all
            pdfs = build_all(str(kb_dir), str(Path(d) / "pdfs"))   # kb_config.json must not break rendering
            self.assertEqual(len(pdfs), 3)


if __name__ == "__main__":
    unittest.main()
