import unittest

import numpy as np

from pasm_v0.diagnose_v06_calibrated import semantic_support_row


class PasmV06CalibratedDiagnosticTests(unittest.TestCase):
    def test_semantic_support_row_reports_direct_teacher_metrics(self) -> None:
        row = semantic_support_row(
            "calibrated",
            np.array([0.0, 1.0, 0.5, 0.5], dtype=np.float32),
            np.array([0.25, 0.75, 0.5, 0.5], dtype=np.float32),
        )

        self.assertEqual(row["model"], "calibrated")
        self.assertAlmostEqual(row["raw_mae"], 0.125)
        self.assertAlmostEqual(row["brier"], 0.03125)
        self.assertIn("effective_patches", row)
        self.assertIn("teacher_effective_patches", row)

if __name__ == "__main__":
    unittest.main()
