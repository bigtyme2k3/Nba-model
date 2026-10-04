import unittest
from build_dashboard import inject_research_link


class DashboardResearchLinkTests(unittest.TestCase):
    def test_dashboard_research_link_is_idempotent(self):
        page="<html><body><main>NBA V1</main></body></html>"
        once=inject_research_link(page)
        self.assertEqual(once,inject_research_link(once))
        self.assertIn("<main>NBA V1</main>",once)
        self.assertIn("No live recommended bets",once)
