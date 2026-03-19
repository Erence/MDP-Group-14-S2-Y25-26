"""
Send a hard-coded sequence of movement and snap commands to the STM32.

Each command is sent one at a time, waiting for an ACK before
sending the next. Edit the MOVEMENTS list below to change the
sequence.

Command reference (STM32):
  FW## / BW##  - Forward / Backward (## = distance/speed value)
  FL## / FR##  - Forward-left / Forward-right turn
  BL## / BR##  - Backward-left / Backward-right turn
  TL## / TR##  - Rotate left / right in place
  SF## / SB##  - Straight forward / backward (alternate)
  RS00         - Gyro reset
  STOP         - Emergency stop

Snap commands (handled by RPi, not sent to STM32):
  SNAP<id>_<signal> - Capture image, send to image-rec API, print result.
                      e.g. SNAP2_C, SNAP4_C

Special:
  FIN          - End of command sequence
"""

import json
import os
import time

import requests

from communication.stm32 import STMLink
from consts import SYMBOL_MAP
from settings import API_IP, API_PORT

# ── Edit this list to change the movement sequence ──────────────
MOVEMENTS = ["SNAP2_C", "FIN"]
# ────────────────────────────────────────────────────────────────

# Max retries for image recognition before giving up
SNAP_MAX_RETRIES = 6


def snap_and_rec(label):
    """
    Capture an image using libcamera-still and send it to the
    image recognition API. Returns the recognized symbol name
    or None if recognition failed.
    """
    url = f"http://{API_IP}:{API_PORT}/image"

    # Read camera config
    con_file = "PiLCConfig9.txt"
    home_user = os.getlogin()
    config_file = f"/home/{home_user}/{con_file}"

    extns = ["jpg", "png", "bmp", "rgb", "yuv420", "raw"]
    shutters = [
        -2000,
        -1600,
        -1250,
        -1000,
        -800,
        -640,
        -500,
        -400,
        -320,
        -288,
        -250,
        -240,
        -200,
        -160,
        -144,
        -125,
        -120,
        -100,
        -96,
        -80,
        -60,
        -50,
        -48,
        -40,
        -30,
        -25,
        -20,
        -15,
        -13,
        -10,
        -8,
        -6,
        -5,
        -4,
        -3,
        0.4,
        0.5,
        0.6,
        0.8,
        1,
        1.1,
        1.2,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        75,
        100,
        112,
        120,
        150,
        200,
        220,
        230,
        239,
        435,
    ]
    meters = ["centre", "spot", "average"]
    awbs = [
        "off",
        "auto",
        "incandescent",
        "tungsten",
        "fluorescent",
        "indoor",
        "daylight",
        "cloudy",
    ]
    denoises = ["off", "cdn_off", "cdn_fast", "cdn_hq"]

    config = []
    with open(config_file, "r") as f:
        for line in f:
            config.append(line.strip())
    config = list(map(int, config))

    speed = config[1]
    gain = config[2]
    brightness = config[3]
    contrast = config[4]
    red = config[6]
    blue = config[7]
    ev = config[8]
    extn = config[15]
    saturation = config[19]
    meter = config[20]
    awb = config[21]
    sharpness = config[22]
    denoise = config[23]
    quality = config[24]

    for retry in range(1, SNAP_MAX_RETRIES + 1):
        shutter = shutters[speed]
        if shutter < 0:
            shutter = abs(1 / shutter)
        sspeed = int(shutter * 1_000_000)
        if (shutter * 1_000_000) - int(shutter * 1_000_000) > 0.5:
            sspeed += 1

        filename = f"{int(time.time())}_{label}_C.jpg"

        cmd = f"libcamera-still -e {extns[extn]} -n -t 100 -o {filename}"
        cmd += f" --brightness {brightness / 100} --contrast {contrast / 100}"
        cmd += f" --shutter {sspeed}"
        if ev != 0:
            cmd += f" --ev {ev}"
        if sspeed > 1_000_000:
            cmd += f" --gain {gain} --immediate"
        else:
            cmd += f" --gain {gain}"
            if awb == 0:
                cmd += f" --awbgains {red / 10},{blue / 10}"
            else:
                cmd += f" --awb {awbs[awb]}"
        cmd += f" --metering {meters[meter]}"
        cmd += f" --saturation {saturation / 10}"
        cmd += f" --sharpness {sharpness / 10}"
        cmd += f" --quality {quality}"
        cmd += f" --denoise {denoises[denoise]}"
        cmd += " --hflip --vflip"

        os.system(cmd)

        response = requests.post(url, files={"file": (filename, open(filename, "rb"))})
        if response.status_code != 200:
            print(f"  [SNAP] ERROR: Image API returned status {response.status_code}")
            return None

        results = json.loads(response.content)
        image_id = results.get("image_id", "NA")
        if isinstance(image_id, list):
            image_id = image_id[0] if image_id else "NA"

        if image_id != "NA":
            symbol = SYMBOL_MAP.get(str(image_id), f"Unknown({image_id})")
            return symbol

        # Retry with adjusted shutter speed
        if retry <= 2:
            print(f"  [SNAP] Attempt {retry}: no match, retrying same settings...")
        elif retry <= 4:
            print(f"  [SNAP] Attempt {retry}: no match, lowering shutter speed...")
            speed -= 1
        elif retry == 5:
            print(f"  [SNAP] Attempt {retry}: no match, raising shutter speed...")
            speed += 3

    print(f"  [SNAP] All {SNAP_MAX_RETRIES} attempts failed, no image recognised.")
    return None


def main():
    stm_link = STMLink()
    stm_link.connect()

    print(f"Running {len(MOVEMENTS)} commands...\n")

    for i, entry in enumerate(MOVEMENTS, 1):
        # ── FIN: end of sequence ──
        if entry == "FIN":
            print(f"[{i}/{len(MOVEMENTS)}] FIN: sequence complete.")
            break

        # ── SNAP command: capture image and print result ──
        if entry.startswith("SNAP"):
            # Format: SNAP<id>_<signal>, e.g. SNAP2_C
            label = entry  # use full command as label for filename
            obstacle_id = entry.replace("SNAP", "").split("_")[0]
            print(
                f"[{i}/{len(MOVEMENTS)}] {entry}: capturing image for obstacle {obstacle_id}..."
            )
            result = snap_and_rec(label)
            print(f"           Result: {result}")
            continue

        # ── STM32 command: send and wait for ACK ──
        print(f"[{i}/{len(MOVEMENTS)}] Sending: {entry}")
        stm_link.send(entry + "\n")
        response = stm_link.recv()
        print(f"           Received: {response}")

    print("\nAll commands complete.")
    stm_link.disconnect()


if __name__ == "__main__":
    main()
