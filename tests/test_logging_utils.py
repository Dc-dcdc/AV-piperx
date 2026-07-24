import logging
import unittest

from lerobot.common.utils.utils import _format_log_record


class LoggingUtilsTest(unittest.TestCase):
    def test_parameterized_log_values_are_interpolated(self):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/tmp/train_pretrain.py",
            lineno=1246,
            msg="已提交异步评估: step=%d, snapshot=%s, pending=%d",
            args=(2000, "/tmp/checkpoint", 1),
            exc_info=None,
        )

        formatted = _format_log_record(record)

        self.assertIn(
            "已提交异步评估: step=2000, "
            "snapshot=/tmp/checkpoint, pending=1",
            formatted,
        )
        self.assertNotIn("%d", formatted)
        self.assertNotIn("%s", formatted)


if __name__ == "__main__":
    unittest.main()
