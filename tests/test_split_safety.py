import unittest

from src.splits import validate_battery_splits


class BatterySplitSafetyTests(unittest.TestCase):
    def test_disjoint_batteries_are_accepted(self):
        validate_battery_splits(["B0005", "B0006"], ["B0018"])

    def test_overlap_is_rejected_even_with_different_casing(self):
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            validate_battery_splits(["B0005"], ["b0005"])

    def test_duplicate_and_empty_groups_are_rejected(self):
        for train, test in ((["B0005", "b0005"], ["B0018"]), ([], ["B0018"])):
            with self.subTest(train=train, test=test), self.assertRaises(ValueError):
                validate_battery_splits(train, test)

    def test_identifier_lists_cannot_be_strings(self):
        with self.assertRaises(ValueError):
            validate_battery_splits("B0005", ["B0018"])


if __name__ == "__main__":
    unittest.main()
