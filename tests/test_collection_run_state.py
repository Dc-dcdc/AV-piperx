import json
import tempfile
import unittest
from pathlib import Path

from data_collect.expert_data_collection.collection_run_state import (
    ExclusiveRunLock,
    atomic_write_json,
)


class AtomicWriteJsonTest(unittest.TestCase):
    def test_atomic_write_replaces_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "metadata.json"

            atomic_write_json(path, {"next_attempt_index": 3})
            atomic_write_json(path, {"next_attempt_index": 4})

            with path.open("r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"next_attempt_index": 4})
            self.assertEqual(list(path.parent.glob(".metadata.json.*.tmp")), [])

    def test_serialization_failure_preserves_previous_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "metadata.json"
            atomic_write_json(path, {"next_attempt_index": 3})

            with self.assertRaises(TypeError):
                atomic_write_json(path, {"not_json_serializable": object()})

            with path.open("r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"next_attempt_index": 3})
            self.assertEqual(list(path.parent.glob(".metadata.json.*.tmp")), [])


class ExclusiveRunLockTest(unittest.TestCase):
    def test_second_writer_is_rejected_until_lock_is_released(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first = ExclusiveRunLock(tmp_dir).acquire()
            second = ExclusiveRunLock(tmp_dir)
            try:
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    second.acquire()
            finally:
                first.close()

            second.acquire()
            second.close()


if __name__ == "__main__":
    unittest.main()
