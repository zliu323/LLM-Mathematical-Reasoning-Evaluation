import unittest

from project import extract_final_answer, is_correct, normalize_answer


class AnswerProcessingTests(unittest.TestCase):
    def test_extracts_nested_box(self):
        self.assertEqual(extract_final_answer(r"Result: \boxed{\frac{3}{4}}"), r"\frac{3}{4}")

    def test_prefers_last_explicit_answer(self):
        text = "Final answer: 10\nCorrection. Final answer: 12"
        self.assertEqual(extract_final_answer(text), "12")

    def test_numeric_equivalence(self):
        self.assertTrue(is_correct(r"\frac{1}{2}", "0.5"))
        self.assertTrue(is_correct("70,000", "70000"))

    def test_tuple_equivalence(self):
        self.assertTrue(is_correct(r"(1, \frac{1}{2})", "(1, 0.5)"))

    def test_latex_normalization(self):
        self.assertEqual(normalize_answer(r"$\left 3.0 \right$"), "3")


if __name__ == "__main__":
    unittest.main()
