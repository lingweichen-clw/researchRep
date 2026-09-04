from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stanchor.diagnostics.transfer_audit import audit_npz_array, audit_edge_csv


class TransferDataAuditTest(unittest.TestCase):
    def test_audit_npz_reports_shape_channels_and_inferred_time_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            values = np.asarray(
                [
                    [[1.0, 2.0], [3.0, 0.0]],
                    [[2.0, 4.0], [5.0, 6.0]],
                ],
                dtype=np.float32,
            )
            np.savez(path, data=values)

            result = audit_npz_array(path)

        self.assertEqual(result["shape"], [2, 2, 2])
        self.assertEqual(result["channels"], 2)
        self.assertEqual(result["timestamp_source"], "inferred_from_row_index")
        self.assertEqual(result["zero_count"], 1)
        self.assertEqual(result["channel_stats"][0]["max"], 5.0)

    def test_audit_edge_csv_reports_node_and_isolated_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edges.csv"
            path.write_text("from,to,cost\n0,1,1.0\n1,0,1.0\n", encoding="utf-8")

            result = audit_edge_csv(path, num_nodes=3)

        self.assertEqual(result["num_nodes"], 3)
        self.assertEqual(result["edge_count"], 2)
        self.assertEqual(result["isolated_nodes"], 1)
        self.assertEqual(result["out_of_range_edges"], 0)


if __name__ == "__main__":
    unittest.main()
