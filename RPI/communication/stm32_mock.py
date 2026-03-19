import os
import re
import time
from multiprocessing import Queue as MPQueue
from typing import Optional

from communication.link import Link

_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}

# Parsed from STM32/STM32_workspace/MDP_STM32/Core/Src/main.c
_STM32_CORE_COMMAND_PATTERNS = (
    re.compile(r"^STOP$"),
    re.compile(r"^(?:FW|FS|BW|BS)(?:\d{2}|--)$"),
    re.compile(r"^(?:FL|FR|BL|BR)(?:\d{2}|--)$"),
    re.compile(r"^(?:TL|TR|IR|DT|ZZ|WX|WN)\d{2}$"),
    re.compile(r"^[AC]\d{3}$"),
)

# Compatibility commands used by RPI workflows even when absent in this STM32 snapshot.
_RPI_COMPAT_COMMAND_PATTERNS = (
    re.compile(r"^RS\d{2}$"),
    re.compile(r"^OB\d{2}$"),
    re.compile(r"^(?:UL|UR|PL|PR)\d{2}$"),
)


def _is_truthy_env(value: str) -> bool:
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def _float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


class MockSTMLink(Link):
    """Protocol-aware STM32 mock implementation."""

    def __init__(self):
        super().__init__()
        self._mock_rx_queue = MPQueue()
        self._mock_strict = _is_truthy_env(os.getenv("RPI_MOCK_STM32_STRICT", ""))
        self._mock_ack_delay = max(0.0, _float_env("RPI_MOCK_STM32_ACK_DELAY", 0.05))
        self._mock_ack_message = os.getenv("RPI_MOCK_STM32_ACK_MESSAGE", "ACK|")

    def connect(self):
        self.logger.info("Connected to mock STM32")

    def disconnect(self):
        self.logger.info("Disconnected from mock STM32")

    def _is_supported_command(self, command: str) -> bool:
        if any(pattern.fullmatch(command) for pattern in _STM32_CORE_COMMAND_PATTERNS):
            return True
        if any(pattern.fullmatch(command) for pattern in _RPI_COMPAT_COMMAND_PATTERNS):
            self.logger.debug(f"[MOCK] Handling RPi compatibility command: {command}")
            return True
        return False

    def _ack_delay_for_command(self, command: str) -> float:
        value = 0
        if len(command) == 4 and command[2:].isdigit():
            value = int(command[2:])
        elif len(command) == 4 and command[1:].isdigit():
            value = int(command[1:])

        if command == "STOP":
            estimated_delay = 0.01
        elif command.startswith(("RS", "TL", "TR", "ZZ")):
            estimated_delay = 0.02
        elif command.startswith(("A", "C")):
            estimated_delay = 0.04 + min(value, 180) * 0.00045
        elif command.startswith(("FW", "FS", "BW", "BS", "OB")):
            estimated_delay = 0.04 + min(value, 99) * 0.002
        elif command.startswith(("FL", "FR", "BL", "BR", "UL", "UR", "PL", "PR")):
            estimated_delay = (
                0.05 if command.endswith("--") else 0.08 + min(value, 30) * 0.003
            )
        elif command.startswith(("DT", "WX", "WN", "IR")):
            estimated_delay = 0.08 + min(value, 99) * 0.002
        else:
            estimated_delay = self._mock_ack_delay

        return max(self._mock_ack_delay, estimated_delay)

    def _queue_ack(self, command: str) -> None:
        ack_delay = self._ack_delay_for_command(command)
        if ack_delay > 0:
            time.sleep(ack_delay)
        self._mock_rx_queue.put(self._mock_ack_message)
        self.logger.debug(
            f"[MOCK] Sent to STM32: {command} | queued {self._mock_ack_message!r} after {ack_delay:.3f}s"
        )

    def send(self, message: str) -> None:
        command = message.strip().upper()
        is_supported = self._is_supported_command(command)

        if not is_supported:
            warning = f"[MOCK] Unsupported STM32 command: {message}"
            if self._mock_strict:
                self.logger.warning(f"{warning} (strict mode on, no ACK sent)")
                return
            self.logger.warning(f"{warning} (strict mode off, sending ACK anyway)")

        self._queue_ack(command)

    def recv(self) -> Optional[str]:
        message = self._mock_rx_queue.get()
        self.logger.debug(f"[MOCK] Received from STM32: {message}")
        return message
