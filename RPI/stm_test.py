"""
Interactive STM32 command sender.
Type a command and press Enter to send it to the STM32.
Press Ctrl+C to quit.
"""

import os
import time
import json
import requests
from communication.stm32 import STMLink
from settings import API_IP, API_PORT


def snap_image(obstacle_id="0", signal="C"):
    """
    Snap an image using the RPi camera (libcamera-still) with settings from PiLCConfig9.txt,
    then send it to the image recognition API.

    :param obstacle_id: The obstacle ID to tag the image with.
    :param signal: The signal direction ('L', 'R', or 'C').
    :return: The image_id result from the API, or 'NA' on failure.
    """
    filename = f"{int(time.time())}_{obstacle_id}_{signal}.jpg"

    con_file = "PiLCConfig9.txt"
    Home_Files = [os.getlogin()]
    config_file = "/home/" + Home_Files[0] + "/" + con_file

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
    with open(config_file, "r") as file:
        for line in file:
            line = line.strip().replace("\x00", "")
            if line and line.lstrip("-").isdigit():
                config.append(int(line))

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

    shutter = shutters[speed]
    if shutter < 0:
        shutter = abs(1 / shutter)
    sspeed = int(shutter * 1_000_000)
    if (shutter * 1_000_000) - int(shutter * 1_000_000) > 0.5:
        sspeed += 1

    rpistr = "libcamera-still -e " + extns[extn] + " -n -t 100 -o " + filename
    rpistr += (
        " --brightness " + str(brightness / 100) + " --contrast " + str(contrast / 100)
    )
    rpistr += " --shutter " + str(sspeed)
    if ev != 0:
        rpistr += " --ev " + str(ev)
    if sspeed > 1_000_000:
        rpistr += " --gain " + str(gain) + " --immediate"
    else:
        rpistr += " --gain " + str(gain)
        if awb == 0:
            rpistr += " --awbgains " + str(red / 10) + "," + str(blue / 10)
        else:
            rpistr += " --awb " + awbs[awb]
    rpistr += " --metering " + meters[meter]
    rpistr += " --saturation " + str(saturation / 10)
    rpistr += " --sharpness " + str(sharpness / 10)
    rpistr += " --quality " + str(quality)
    rpistr += " --denoise " + denoises[denoise]
    rpistr += " --hflip --vflip"

    print(f"  Capturing image: {filename}")
    os.system(rpistr)

    # Send to image recognition API
    url = f"http://{API_IP}:{API_PORT}/image"
    try:
        with open(filename, "rb") as f:
            response = requests.post(url, files={"file": (filename, f)})
        if response.status_code != 200:
            print(f"  API error: status {response.status_code}")
            return "NA"
        results = json.loads(response.content)
        print(f"  API response: {results}")
        image_id = results.get("image_id", "NA")
        if isinstance(image_id, list):
            image_id = str(image_id[0]) if image_id else "NA"
        else:
            image_id = str(image_id)
        print(f"  Image recognition result: {image_id}")
        return image_id
    except Exception as e:
        print(f"  Error sending to API: {e}")
        return "NA"


def main():
    stm_link = STMLink()
    stm_link.connect()

    print("Connected. Type commands and press Enter to send. Ctrl+C to quit.")
    print("  Type SNAP to capture an image and run image recognition.\n")

    try:
        while True:
            cmd = input("STM> ")
            if not cmd:
                continue
            if cmd.strip().upper().startswith("SNAP"):
                snap_image()
                continue
            stm_link.send(cmd + "\n")
            print(f"  Sent: {cmd}")
            response = stm_link.recv()
            print(f"  Recv: {response}")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
    finally:
        stm_link.disconnect()


if __name__ == "__main__":
    main()
