import unittest

from clv_reconciler import line_clv, select_close, match_key


class TestClvDirection(unittest.TestCase):
    def test_over_better_entry_is_positive(self):
        self.assertEqual(line_clv(17.5, 18.5, "OVER"), 1.0)

    def test_over_worse_entry_is_negative(self):
        self.assertEqual(line_clv(18.5, 17.5, "OVER"), -1.0)

    def test_under_better_entry_is_positive(self):
        self.assertEqual(line_clv(18.5, 17.5, "UNDER"), 1.0)

    def test_under_worse_entry_is_negative(self):
        self.assertEqual(line_clv(17.5, 18.5, "UNDER"), -1.0)

    def test_selects_closest_valid_pretip_snapshot(self):
        rows=[
            {"minutes_to_tip":22.0,"line":17.5},
            {"minutes_to_tip":8.0,"line":18.5},
            {"minutes_to_tip":-1.0,"line":19.5},
        ]
        self.assertEqual(select_close(rows)["line"],18.5)

    def test_exact_book_is_part_of_match_key(self):
        base={
            "date":"2026-10-27",
            "game":"A @ B",
            "player":"Player One",
            "stat":"PTS",
            "side":"OVER",
            "sportsbook_key":"fanduel",
        }
        other=dict(base)
        other["sportsbook_key"]="draftkings"
        self.assertNotEqual(match_key(base),match_key(other))


if __name__=="__main__":
    unittest.main()
