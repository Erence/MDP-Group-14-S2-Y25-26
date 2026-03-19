#!/usr/bin/env python3
import json
import queue
import time
from multiprocessing import Process, Manager
from typing import Optional
import os
import requests
from communication.android import AndroidLink, AndroidMessage
from communication.stm32 import STMLink
from consts import SYMBOL_MAP
from logger import prepare_logger
from settings import API_IP, API_PORT


class RaspberryPi:
    """
    Week 9 - Fastest Car Task

    Simplified flow where STM32 handles all movement logic.
    RPi only decides left/right at each obstacle via image recognition.

    Protocol:
        1. RPi sends "START\n" upon receiving start from Android
        2. STM32 drives to obstacle 1, sends "SNAP1"
        3. RPi captures image, recognizes direction, sends "LEFT1" or "RIGHT1"
        4. STM32 navigates obstacle 1, drives to obstacle 2, sends "SNAP2"
        5. RPi captures image, recognizes direction, sends "LEFT2" or "RIGHT2"
        6. STM32 navigates obstacle 2, returns to carpark, sends "DONE"
    """

    def __init__(self):
        self.logger = prepare_logger()
        self.android_link = AndroidLink()
        self.stm_link = STMLink()

        self.manager = Manager()

        # Events
        self.android_dropped = self.manager.Event()
        self.stm_snap1 = self.manager.Event()
        self.stm_snap2 = self.manager.Event()
        self.stm_done = self.manager.Event()

        # Queues
        self.android_queue = self.manager.Queue()

        # Processes
        self.proc_recv_android = None
        self.proc_recv_stm32 = None
        self.proc_android_sender = None
        self.proc_task_runner = None

    def start(self):
        """Starts the RPi orchestrator"""
        try:
            self.android_link.connect()
            self.android_queue.put(
                AndroidMessage("info", "You are connected to the RPi!")
            )

            self.stm_link.connect()
            self.check_api()

            self.proc_recv_android = Process(target=self.recv_android)
            self.proc_recv_stm32 = Process(target=self.recv_stm)
            self.proc_android_sender = Process(target=self.android_sender)

            self.proc_recv_android.start()
            self.proc_recv_stm32.start()
            self.proc_android_sender.start()

            self.logger.info("Child Processes started")

            self.android_queue.put(AndroidMessage("info", "Robot is ready!"))
            self.android_queue.put(AndroidMessage("mode", "path"))

            self.reconnect_android()

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stops all processes on the RPi and disconnects gracefully"""
        self.android_link.disconnect()
        self.stm_link.disconnect()
        self.logger.info("Program exited!")

    def reconnect_android(self):
        """Handles the reconnection to Android in the event of a lost connection."""
        self.logger.info("Reconnection handler is watching...")

        while True:
            self.android_dropped.wait()

            self.logger.error("Android link is down!")

            self.logger.debug("Killing android child processes")
            self.proc_android_sender.kill()
            self.proc_recv_android.kill()

            self.proc_android_sender.join()
            self.proc_recv_android.join()
            assert self.proc_android_sender.is_alive() is False
            assert self.proc_recv_android.is_alive() is False
            self.logger.debug("Android child processes killed")

            self.android_link.disconnect()
            self.android_link.connect()

            self.proc_recv_android = Process(target=self.recv_android)
            self.proc_android_sender = Process(target=self.android_sender)

            self.proc_recv_android.start()
            self.proc_android_sender.start()

            self.logger.info("Android child processes restarted")
            self.android_queue.put(AndroidMessage("info", "You are reconnected!"))
            self.android_queue.put(AndroidMessage("mode", "path"))

            self.android_dropped.clear()

    def recv_android(self) -> None:
        """[Child Process] Processes the messages received from Android"""
        while True:
            msg_str: Optional[str] = None
            try:
                msg_str = self.android_link.recv()
            except OSError:
                self.android_dropped.set()
                self.logger.debug("Event set: Android connection dropped")

            if msg_str is None:
                continue

            message: dict = json.loads(msg_str)

            if message["cat"] == "control":
                if message["value"] == "start":
                    if not self.check_api():
                        self.logger.error("API is down! Start command aborted.")
                        continue

                    self.logger.info(
                        "Start command received, starting robot on Week 9 task!"
                    )
                    self.android_queue.put(AndroidMessage("status", "running"))

                    # Launch the task runner in a separate process
                    if self.proc_task_runner and self.proc_task_runner.is_alive():
                        self.proc_task_runner.kill()
                        self.proc_task_runner.join()
                    self.proc_task_runner = Process(target=self.run_task)
                    self.proc_task_runner.start()

                elif message["value"] == "stop":
                    self.stm_link.send("FIN")
                    self.logger.info("Stop command received, sent FIN to STM32.")

    def recv_stm(self) -> None:
        """[Child Process] Receive messages from STM32"""
        while True:
            message: str = self.stm_link.recv()

            if message.startswith("ACK"):
                self.logger.debug(f"ACK from STM32 received")
            elif message.startswith("SNAP1"):
                self.logger.info("SNAP1 from STM32 received")
                self.stm_snap1.set()
            elif message.startswith("SNAP2"):
                self.logger.info("SNAP2 from STM32 received")
                self.stm_snap2.set()
            elif message.startswith("DONE"):
                self.logger.info("DONE from STM32 received")
                self.stm_done.set()
            else:
                self.logger.warning(f"Ignored unknown message from STM: {message}")

    def run_task(self) -> None:
        """
        [Child Process] Main task runner for Week 9 fastest car.

        Sequential flow:
            1. Send "START\n" to STM32
            2. Wait for "SNAP1" -> snap image -> send "LEFT1" or "RIGHT1"
            3. Wait for "SNAP2" -> snap image -> send "LEFT2" or "RIGHT2"
            4. Wait for "DONE" (robot back in carpark)
        """
        # Clear all events
        self.stm_snap1.clear()
        self.stm_snap2.clear()
        self.stm_done.clear()

        # Step 1: Send START to STM32
        self.logger.info("Step 1: Sending START to STM32")
        self.android_queue.put(AndroidMessage("status", "moving"))
        self.stm_link.send("START\n")

        # Step 2: Wait for SNAP1, snap obstacle 1, send direction
        self.logger.info("Step 2: Waiting for SNAP1 from STM32")
        self.stm_snap1.wait()
        self.stm_snap1.clear()
        self.logger.info("Step 2: Snapping obstacle 1")
        direction = self.snap_and_rec("Obstacle_1")
        self.logger.info(f"Step 2: Obstacle 1 direction: {direction}")

        if direction == "LEFT":
            cmd = "LEFT"
        elif direction == "RIGHT":
            cmd = "RIGHT"
        else:
            cmd = "LEFT"  # Default to left if recognition fails
            self.logger.warning(
                f"Step 2: Recognition returned '{direction}', defaulting to LEFT"
            )

        self.logger.info(f"Step 2: Sending {cmd} to STM32")
        self.stm_link.send(cmd)

        # Step 3: Wait for SNAP2, snap obstacle 2, send direction
        self.logger.info("Step 3: Waiting for SNAP2 from STM32")
        self.stm_snap2.wait()
        self.stm_snap2.clear()
        self.logger.info("Step 3: Snapping obstacle 2")
        direction = self.snap_and_rec("Obstacle_2")
        self.logger.info(f"Step 3: Obstacle 2 direction: {direction}")

        if direction == "LEFT":
            cmd = "LEFT"
        elif direction == "RIGHT":
            cmd = "RIGHT"
        else:
            cmd = "RIGHT"  # Default to right if recognition fails
            self.logger.warning(
                f"Step 3: Recognition returned '{direction}', defaulting to RIGHT"
            )

        self.logger.info(f"Step 3: Sending {cmd} to STM32")
        self.stm_link.send(cmd)

        # Step 4: Wait for DONE
        self.logger.info("Step 4: Waiting for DONE from STM32")
        self.stm_done.wait()
        self.stm_done.clear()
        self.logger.info("Step 4: Robot back in carpark, task complete!")

        # Task complete
        self.android_queue.put(AndroidMessage("info", "Task complete!"))
        self.android_queue.put(AndroidMessage("status", "finished"))
        self.request_stitch()

    def android_sender(self) -> None:
        """[Child Process] Sends queued messages to Android"""
        while True:
            try:
                message: AndroidMessage = self.android_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self.android_link.send(message)
            except OSError:
                self.android_dropped.set()
                self.logger.debug("Event set: Android dropped")

    def snap_and_rec(self, obstacle_id: str) -> Optional[str]:
        """
        RPi snaps an image and calls the API for image-rec.
        :param obstacle_id: label for logging/filename
        :return: recognized symbol name (e.g. "Left Arrow", "Right Arrow") or None
        """
        self.logger.info(f"Capturing image for obstacle: {obstacle_id}")
        url = f"http://{API_IP}:{API_PORT}/image"
        filename = f"{int(time.time())}_{obstacle_id}.jpg"

        con_file = "PiLCConfig9.txt"
        Home_Files = []
        Home_Files.append(os.getlogin())
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
            line = file.readline()
            while line:
                config.append(line.strip())
                line = file.readline()
            config = list(map(int, config))
        mode = config[0]
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

        retry_count = 0

        while True:
            retry_count += 1

            shutter = shutters[speed]
            if shutter < 0:
                shutter = abs(1 / shutter)
            sspeed = int(shutter * 1000000)
            if (shutter * 1000000) - int(shutter * 1000000) > 0.5:
                sspeed += 1

            rpistr = "libcamera-still -e " + extns[extn] + " -n -t 100 -o " + filename
            rpistr += (
                " --brightness "
                + str(brightness / 100)
                + " --contrast "
                + str(contrast / 100)
            )
            rpistr += " --shutter " + str(sspeed)
            if ev != 0:
                rpistr += " --ev " + str(ev)
            if sspeed > 1000000 and mode == 0:
                rpistr += " --gain " + str(gain) + " --immediate "
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

            os.system(rpistr)

            self.logger.debug("Requesting from image API")

            response = requests.post(
                url, files={"file": (filename, open(filename, "rb"))}
            )

            if response.status_code != 200:
                self.logger.error(
                    "Something went wrong when requesting from image-rec API."
                )
                return None

            results = json.loads(response.content)

            if results["image_id"] != "NA" or retry_count > 6:
                break
            elif retry_count <= 2:
                self.logger.info(f"Image recognition results: {results}")
                self.logger.info("Recapturing with same shutter speed...")
            elif retry_count <= 4:
                self.logger.info(f"Image recognition results: {results}")
                self.logger.info("Recapturing with lower shutter speed...")
                speed -= 1
            elif retry_count == 5:
                self.logger.info(f"Image recognition results: {results}")
                self.logger.info("Recapturing with lower shutter speed...")
                speed += 3

        ans = SYMBOL_MAP.get(results["image_id"])
        self.logger.info(f"Image recognition results: {results} ({ans})")
        return ans

    def request_stitch(self):
        url = f"http://{API_IP}:{API_PORT}/stitch"
        response = requests.get(url)
        if response.status_code != 200:
            self.logger.error(
                "Something went wrong when requesting stitch from the API."
            )
            return
        self.logger.info("Images stitched!")

    def check_api(self) -> bool:
        url = f"http://{API_IP}:{API_PORT}/status"
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                self.logger.debug("API is up!")
                return True
        except ConnectionError:
            self.logger.warning("API Connection Error")
            return False
        except requests.Timeout:
            self.logger.warning("API Timeout")
            return False
        except Exception as e:
            self.logger.warning(f"API Exception: {e}")
            return False


if __name__ == "__main__":
    rpi = RaspberryPi()
    rpi.start()
