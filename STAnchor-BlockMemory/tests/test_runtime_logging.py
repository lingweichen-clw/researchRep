"""Runtime observability contracts."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from stanchor.utils import create_run_logger


def close_logger(logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


class RuntimeLoggingTest(unittest.TestCase):
    def test_run_logger_writes_to_console_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "train.log"
            console = io.StringIO()
            with contextlib.redirect_stderr(console):
                logger = create_run_logger("stanchor.test.runtime", log_path)
                logger.info("trainable_parameters=123")
                for handler in logger.handlers:
                    handler.flush()

            self.assertIn("trainable_parameters=123", console.getvalue())
            self.assertIn("trainable_parameters=123", log_path.read_text(encoding="utf-8"))
            close_logger(logger)

    def test_run_logger_replaces_handlers_without_duplicate_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "train.log"
            logger = create_run_logger("stanchor.test.reconfigure", log_path)
            logger = create_run_logger("stanchor.test.reconfigure", log_path)
            logger.info("one-line-only")
            for handler in logger.handlers:
                handler.flush()

            self.assertEqual(
                log_path.read_text(encoding="utf-8").count("one-line-only"),
                1,
            )
            close_logger(logger)


if __name__ == "__main__":
    unittest.main()
