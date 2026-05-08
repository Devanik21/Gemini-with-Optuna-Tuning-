"""
Tests for configuration settings.
"""
import unittest
from src.config.settings import settings

class TestConfig(unittest.TestCase):
    def test_default_task(self):
        self.assertIsNotNone(settings.DEFAULT_TASK)
