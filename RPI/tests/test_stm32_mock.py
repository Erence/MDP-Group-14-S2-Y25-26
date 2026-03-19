import os
import queue
import time
import unittest

from communication.stm32 import STMLink


class TestSTMLinkMock(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = {
            "RPI_FORCE_MOCK_STM32": os.getenv("RPI_FORCE_MOCK_STM32"),
            "RPI_MOCK_STM32_ACK_DELAY": os.getenv("RPI_MOCK_STM32_ACK_DELAY"),
            "RPI_MOCK_STM32_STRICT": os.getenv("RPI_MOCK_STM32_STRICT"),
        }
        os.environ["RPI_FORCE_MOCK_STM32"] = "1"
        os.environ["RPI_MOCK_STM32_ACK_DELAY"] = "0"
        os.environ["RPI_MOCK_STM32_STRICT"] = "0"

    def tearDown(self) -> None:
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _new_link(self) -> STMLink:
        link = STMLink()
        link.connect()
        self.assertTrue(link.is_mock)
        return link

    def test_core_command_ack(self) -> None:
        link = self._new_link()
        link.send("FW10")
        self.assertEqual("ACK|", link.recv())
        link.disconnect()

    def test_rpi_compat_command_ack(self) -> None:
        link = self._new_link()
        link.send("RS00")
        self.assertEqual("ACK|", link.recv())
        link.disconnect()

    def test_unsupported_command_ack_in_non_strict_mode(self) -> None:
        link = self._new_link()
        link.send("XX00")
        self.assertEqual("ACK|", link.recv())
        link.disconnect()

    def test_unsupported_command_no_ack_in_strict_mode(self) -> None:
        os.environ["RPI_MOCK_STM32_STRICT"] = "1"
        link = self._new_link()

        link.send("XX00")
        time.sleep(0.05)

        self.assertIsNotNone(link._mock_link)
        with self.assertRaises(queue.Empty):
            link._mock_link._mock_rx_queue.get_nowait()

        link.disconnect()


if __name__ == "__main__":
    unittest.main()
