"""
Tests for evaluation metrics.
"""
import unittest
from src.evaluation.metrics import rouge_l, exact_match

class TestMetrics(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(exact_match("hello", "hello"), 1.0)
        self.assertEqual(exact_match("hello", "world"), 0.0)

    def test_rouge_l(self):
        self.assertTrue(rouge_l("hello world", "hello") > 0)
