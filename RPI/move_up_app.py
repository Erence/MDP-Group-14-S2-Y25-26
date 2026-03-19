#!/usr/bin/env python3
"""
Simple app that listens for a "move:up" Bluetooth message from Android
and sends SF100 (move forward 100) to STM32.

Expected Android message format:
    {"cat": "move", "value": "up"}
"""

import json
from typing import Optional

from communication.android import AndroidLink, AndroidMessage
from communication.stm32 import STMLink
from logger import prepare_logger


def main():
    logger = prepare_logger()

    # --- Connect to Android and STM32 ---
    stm_link = STMLink()

    logger.info("Connecting to STM32...")
    stm_link.connect()
    logger.info("STM32 connected.")
    stm_link.send("SF100")


if __name__ == "__main__":
    main()
