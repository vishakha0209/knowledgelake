import unittest

from knowledgelake.classify import classify_all, load_config
from knowledgelake.models import Document
from knowledgelake.parse import parse_document

NOTES = """Team notes
1. What is the difference between WHERE and HAVING?
Answer: WHERE filters rows before grouping; HAVING filters groups.

2. Find the second highest salary.
SELECT MAX(salary) FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

What is the difference between a linked service and a dataset in ADF?
How would you design a metadata-driven ADF pipeline for 100 tables?
"""


class ParseClassifyTests(unittest.TestCase):
    def setUp(self):
        self.items = parse_document(Document("notes.txt", NOTES, "plain-text"))

    def test_finds_numbered_and_bare_questions(self):
        self.assertEqual(len(self.items), 4)
        self.assertEqual(self.items[2].a, "")  # question-only item

    def test_prose_answer_not_fenced_but_sql_is(self):
        self.assertFalse(self.items[0].a.startswith("```"))
        self.assertTrue(self.items[1].a.startswith("```"))

    def test_topics(self):
        classify_all(self.items, load_config())
        self.assertEqual(self.items[0].topic, "SQL")
        self.assertEqual(self.items[3].topic, "Azure Data Factory")


if __name__ == "__main__":
    unittest.main()
