import os
from typing import Optional

from communication.link import Link
from communication.stm32_mock import MockSTMLink
from settings import BAUD_RATE, SERIAL_PORT

try:
    import serial  # type: ignore
except ImportError:
    serial = None


def _is_truthy_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


class STMLink(Link):
    """UART link for STM32 with automatic mock fallback."""

    def __init__(self):
        super().__init__()
        self.serial_link = None
        self._mock_link: Optional[MockSTMLink] = None
        self._force_mock = _is_truthy_env(os.getenv("RPI_FORCE_MOCK_STM32", ""))

    @property
    def is_mock(self) -> bool:
        return self._mock_link is not None

    def _enable_mock(self, reason: str) -> None:
        self._mock_link = MockSTMLink()
        self.logger.warning(f"{reason} Using mock STM32 link.")
        self._mock_link.connect()

    def connect(self):
        """Connect to STM32 using serial UART, or fallback to mock mode."""
        if self._force_mock:
            self._enable_mock("RPI_FORCE_MOCK_STM32 is enabled.")
            return

        if serial is None:
            self._enable_mock("pyserial is not installed.")
            return

        try:
            self.serial_link = serial.Serial(SERIAL_PORT, BAUD_RATE)
            self._mock_link = None
            self.logger.info("Connected to STM32")
        except Exception as exc:
            self.serial_link = None
            self._enable_mock(
                f"Unable to connect to STM32 on {SERIAL_PORT} at {BAUD_RATE} baud: {exc}."
            )

    def disconnect(self):
        """Disconnect from STM32 or mock backend."""
        if self._mock_link is not None:
            self._mock_link.disconnect()
            self._mock_link = None
            return

        if self.serial_link is not None:
            self.serial_link.close()
            self.serial_link = None
            self.logger.info("Disconnected from STM32")

    def send(self, message: str) -> None:
        """Send message to STM32."""
        if self._mock_link is not None:
            self._mock_link.send(message)
            return

        if self.serial_link is None:
            raise ConnectionError("STM32 serial link is not connected.")

        self.serial_link.write(f"{message}".encode("utf-8"))
        self.logger.debug(f"Sent to STM32: {message}")

    def recv(self) -> Optional[str]:
        """Receive message from STM32."""
        if self._mock_link is not None:
            return self._mock_link.recv()

        if self.serial_link is None:
            raise ConnectionError("STM32 serial link is not connected.")

        message = self.serial_link.readline().strip().decode("utf-8")
        self.logger.debug(f"Received from STM32: {message}")
        return message
