from unittest import TestCase

from transcriber.benchmark import word_error_rate


class WordErrorRateTests(TestCase):
    def test_identical_text_has_zero_error(self):
        self.assertEqual(word_error_rate("Hello, world!", "hello world"), 0.0)

    def test_counts_substitution(self):
        self.assertEqual(word_error_rate("one two three", "one four three"), 1 / 3)

    def test_empty_reference(self):
        self.assertEqual(word_error_rate("", ""), 0.0)
        self.assertEqual(word_error_rate("", "unexpected"), 1.0)
