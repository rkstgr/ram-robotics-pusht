from __future__ import annotations

import unittest

from scripts.train_ram_pusht import safe_rate


class TrainingMetricsTests(unittest.TestCase):
    def test_safe_rate(self) -> None:
        self.assertEqual(safe_rate(10, 2), 5.0)
        self.assertEqual(safe_rate(10, 0), 0.0)
        self.assertEqual(safe_rate(10, -1), 0.0)


if __name__ == "__main__":
    unittest.main()
