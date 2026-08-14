"""验证 v0.5 稠密融合选点诊断的空间统计。"""

import unittest

import numpy as np

from pasm_v0.diagnose_v05_selection import nms_peak_mask, selection_grid_counts


class PasmV05SelectionDiagnosticTests(unittest.TestCase):
    def test_grid_counts_separate_dense_roof_like_cell_from_selected_points(self) -> None:
        """诊断应同时报告每格 NMS 候选数和最终入选数。"""
        peaks = np.array(
            [[True, False, False, False], [False, True, False, False],
             [False, False, False, True], [False, False, False, False]],
            dtype=bool,
        )
        points = np.array([[0.5, 0.5], [1.5, 1.5], [3.5, 2.5]], dtype=np.float32)

        candidates, selected = selection_grid_counts(peaks, points, grid_size=2)

        np.testing.assert_array_equal(candidates, np.array([[2, 0], [0, 1]]))
        np.testing.assert_array_equal(selected, np.array([[2, 0], [0, 1]]))

    def test_nms_peak_mask_keeps_only_local_maxima(self) -> None:
        """3×3 NMS 候选掩码应排除被邻居超过的普通像素。"""
        scores = np.array(
            [[0.1, 0.2, 0.1], [0.2, 1.0, 0.2], [0.1, 0.2, 0.1]], dtype=np.float32
        )

        peaks = nms_peak_mask(scores, nms_size=3)

        np.testing.assert_array_equal(
            peaks, np.array([[False, False, False], [False, True, False], [False, False, False]])
        )


if __name__ == "__main__":
    unittest.main()
