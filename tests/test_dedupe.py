import unittest

from knowledgelake.dedupe import deduplicate, is_duplicate
from knowledgelake.models import QAItem


class DedupeTests(unittest.TestCase):
    def test_reworded_question_is_duplicate(self):
        self.assertTrue(is_duplicate("Find the second highest salary from the Employee table.",
                                     "Retrieve the second highest salary from the employees table"))

    def test_meaning_flip_is_not_duplicate(self):
        self.assertFalse(is_duplicate("Find the second highest salary from the Employee table.",
                                      "Find the second lowest salary from the Employee table."))
        self.assertFalse(is_duplicate("Find employees hired before their managers.",
                                      "Find employees hired after their managers."))
        self.assertFalse(is_duplicate("Find employees promoted more than once.",
                                      "Find employees promoted more than twice."))

    def test_merge_keeps_best_answer_and_all_sources(self):
        items = [QAItem(q="What is a broadcast join?", a="", sources=["a.pdf"]),
                 QAItem(q="What is a broadcast join in Spark?", a="Small table copied to every executor.",
                        sources=["b.pdf"])]
        merged, removed = deduplicate(items)
        self.assertEqual(removed, 1)
        self.assertEqual(merged[0].a, "Small table copied to every executor.")
        self.assertEqual(sorted(merged[0].sources), ["a.pdf", "b.pdf"])


if __name__ == "__main__":
    unittest.main()
