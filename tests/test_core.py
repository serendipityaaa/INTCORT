import unittest

from intcort.parsing import confidence_vote, parse_relation, route_binary, route_unary
from intcort.transforms import mapped, reverse_question


class CoreTests(unittest.TestCase):
    def test_reverse_question_preserves_suffix(self):
        question = "Where is the mug in relation to the knife? Answer with left or right."
        self.assertEqual(
            reverse_question(question).text,
            "Where is the knife in relation to the mug? Answer with left or right.",
        )

    def test_inverse_mappings(self):
        self.assertEqual(mapped("left", {"left": "right"}), "right")
        self.assertEqual(mapped("above", {"above": "below"}), "below")

    def test_relation_parser(self):
        self.assertEqual(
            parse_relation("The answer is right.", ["left", "right"]),
            "right",
        )

    def test_confidence_vote(self):
        self.assertEqual(
            confidence_vote(["left", "right", "left"], [-1.0, -1.5, -1.0], None),
            "left",
        )

    def test_original_routing_rules(self):
        self.assertEqual(route_unary("left", "right", "left", -1, -1, -1), "left")
        self.assertEqual(route_unary("top", "top", "bottom", -1, -1, -1), "top")
        self.assertEqual(route_binary("front", "front", "behind", -1, -1, -1), "behind")


if __name__ == "__main__":
    unittest.main()
