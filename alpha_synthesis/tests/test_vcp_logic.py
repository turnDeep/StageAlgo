import unittest
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vcp_screener import find_vcp_patterns

class TestVCPLogic(unittest.TestCase):
    def setUp(self):
        self.base_date = datetime(2025, 1, 1)

    def create_pivots(self, prices):
        """Helper to create pivots list from a list of prices.
        Assumes alternating High/Low starting with High.
        """
        pivots = []
        for i, p in enumerate(prices):
            pivots.append({
                'idx': i,
                'date': self.base_date + timedelta(days=i*10),
                'price': p,
                'type': 'high' if i % 2 == 0 else 'low'
            })
        return pivots

    def test_2_contractions(self):
        # 3 waves:
        # 1. H=100 -> L=80 (20%)
        # 2. H=90 -> L=81 (10%)
        # 3. H=85 -> L=80.75 (5%)
        # Check depths logic: (High-Low)/High

        # Wave 1: 100 -> 80. Depth = 20/100 = 0.20
        # Wave 2: 90 -> 81. Depth = 9/90 = 0.10
        # Wave 3: 85 -> 80.75. Depth = 4.25/85 = 0.05

        prices = [100, 80, 90, 81, 85, 80.75]
        pivots = self.create_pivots(prices)

        patterns = find_vcp_patterns(None, pivots) # df is unused in logic

        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0]['type'], '2_Contractions')
        self.assertEqual(patterns[0]['depths'], [0.20, 0.10, 0.05])

    def test_3_contractions(self):
        # 4 waves decreasing: 30% -> 15% -> 8% -> 4%
        # W1: 100 -> 70 (30%)
        # W2: 80 -> 68 (12/80 = 15%)
        # W3: 75 -> 69 (6/75 = 8%)
        # W4: 72 -> 69.12 (2.88/72 = 4%)

        prices = [100, 70, 80, 68, 75, 69, 72, 69.12]
        pivots = self.create_pivots(prices)

        patterns = find_vcp_patterns(None, pivots)

        # Should find 3 contractions
        # Note: 3 contractions logic might also satisfy 2 contractions for the last 3 waves.
        # But my code uses 'continue' if 3 contractions found.

        found_3 = [p for p in patterns if p['type'] == '3_Contractions']
        self.assertTrue(len(found_3) >= 1)
        # Check almost equal for depths
        expected = [0.30, 0.15, 0.08, 0.04]
        for i, val in enumerate(found_3[0]['depths']):
            self.assertAlmostEqual(val, expected[i], places=5)

    def test_no_contraction(self):
        # Expansion
        # 10% -> 20%
        prices = [100, 90, 95, 76] # 10/100=10%, 19/95=20%
        pivots = self.create_pivots(prices)

        patterns = find_vcp_patterns(None, pivots)
        self.assertEqual(len(patterns), 0)

if __name__ == '__main__':
    unittest.main()
