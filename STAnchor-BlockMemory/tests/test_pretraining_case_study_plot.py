from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from stanchor.diagnostics.pretraining_curves import load_pretraining_history


class PretrainingCaseStudyPlotTest(unittest.TestCase):
    def test_load_pretraining_history_extracts_joint_losses(self) -> None:
        record = {
            "epoch": 3,
            "train": {"total": 0.5, "reconstruction": 0.2, "retrieval": 3.0},
            "val": {
                "total": 0.4,
                "reconstruction": 0.15,
                "retrieval": 2.5,
                "teacher_effective_support": 4.0,
                "student_effective_support": 8.0,
            },
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            history = load_pretraining_history(path)

        self.assertEqual(history["epoch"], [3])
        self.assertEqual(history["train_total"], [0.5])
        self.assertEqual(history["val_reconstruction"], [0.15])
        self.assertEqual(history["val_retrieval"], [2.5])
        self.assertEqual(history["teacher_keff"], [4.0])
        self.assertEqual(history["student_keff"], [8.0])

    def test_load_pretraining_history_keeps_unvalidated_epochs_as_nan(self) -> None:
        records = [
            {
                "epoch": 1,
                "train": {"total": 0.5, "reconstruction": 0.2, "retrieval": 3.0},
                "val": None,
            },
            {
                "epoch": 2,
                "train": {"total": 0.4, "reconstruction": 0.1, "retrieval": 2.5},
                "val": {
                    "total": 0.35,
                    "reconstruction": 0.12,
                    "retrieval": 2.2,
                    "teacher_effective_support": 4.0,
                    "student_effective_support": 8.0,
                },
            },
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            history = load_pretraining_history(path)

        self.assertTrue(history["val_total"][0] != history["val_total"][0])
        self.assertEqual(history["val_total"][1], 0.35)


if __name__ == "__main__":
    unittest.main()
