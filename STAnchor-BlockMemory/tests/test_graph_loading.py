from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stanchor.data.graph import load_dense_adjacency


class GraphLoadingTest(unittest.TestCase):
    def test_protocol_zero_pickle_recovers_from_crlf_conversion(self) -> None:
        adjacency = np.array([[0.0, 0.5], [0.25, 0.0]], dtype=np.float32)
        payload = pickle.dumps(
            [["node-0", "node-1"], {"node-0": 0, "node-1": 1}, adjacency],
            protocol=0,
        )
        converted = payload.replace(b"\n", b"\r\n")
        self.assertNotEqual(payload, converted)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adj_mx.pkl"
            path.write_bytes(converted)

            loaded = load_dense_adjacency(path)

        np.testing.assert_allclose(loaded, adjacency)

    def test_invalid_pickle_is_not_silently_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adj_mx.pkl"
            path.write_bytes(b"this is not a pickle\r\n")

            with self.assertRaises(pickle.UnpicklingError):
                load_dense_adjacency(path)


if __name__ == "__main__":
    unittest.main()
