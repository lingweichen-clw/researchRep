from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stanchor.data.dataset import load_npz_series
from stanchor.data.graph import load_graph


class TransferDataLoadingTest(unittest.TestCase):
    def test_npz_loader_selects_speed_channel_and_builds_five_minute_calendar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traffic.npz"
            values = np.zeros((577, 2, 3), dtype=np.float32)
            values[..., 2] = 60.0
            np.savez(path, data=values)

            series = load_npz_series(
                path,
                channel_index=2,
                start_weekday=2,
                start_slot=287,
            )

        self.assertEqual(series.values.shape, (577, 2, 1))
        self.assertEqual(series.num_channels, 1)
        self.assertEqual(series.slots_per_day, 288)
        self.assertEqual(int(series.slot[0]), 287)
        self.assertEqual(int(series.slot[1]), 0)
        self.assertEqual(int(series.weekday[0]), 2)
        self.assertEqual(int(series.weekday[1]), 3)
        self.assertEqual(int(np.diff(series.timestamps_ns).min()), 5 * 60 * 1_000_000_000)

    def test_csv_graph_adds_self_loops_and_honors_explicit_node_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edges.csv"
            path.write_text(
                "from,to,cost\n0,1,10.0\n1,0,10.0\n",
                encoding="utf-8",
            )

            graph = load_graph(path, num_nodes=3)

        self.assertEqual(graph.num_nodes, 3)
        graph.validate()
        self.assertEqual(int((graph.edge_index[0] == graph.edge_index[1]).sum()), 3)
        self.assertTrue(bool((graph.edge_weight > 0).all()))


if __name__ == "__main__":
    unittest.main()
