import os
import shutil
import time
import glob
import torch
from PIL import Image
import cv2
import random
import string
import numpy as np
import random
from ultralytics import YOLO


def get_random_string(length):
    """
    Generate a random string of fixed length

    Inputs
    ------
    length: int - length of the string to be generated

    Returns
    -------
    str - random string

    """
    result_str = "".join(random.choice(string.ascii_letters) for i in range(length))
    return result_str


def load_model():
    """
    Load the model from the local directory
    """
    # model = torch.hub.load('./', 'custom', path='YOLOv5_new.pt', source='local')
    model = YOLO("seg_v4.pt")
    return model


def draw_all_bboxes(img, pred_list, chosen_pred=None):
    """
    Draw ALL detected bounding boxes on the image and save to own_results.
    The chosen prediction is highlighted in green, others in red.

    Inputs
    ------
    img: numpy.ndarray - image (RGB)
    pred_list: list - list of prediction dicts with xmin, ymin, xmax, ymax, confidence, name
    chosen_pred: dict or None - the selected prediction to highlight
    """
    rand = str(int(time.time()))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for pred in pred_list:
        x1, y1, x2, y2 = (
            int(pred["xmin"]),
            int(pred["ymin"]),
            int(pred["xmax"]),
            int(pred["ymax"]),
        )
        label = f"{pred['name']} {pred['confidence']:.2f}"

        # Green for chosen prediction, red for others
        if (
            chosen_pred
            and pred["xmin"] == chosen_pred["xmin"]
            and pred["ymin"] == chosen_pred["ymin"]
        ):
            color = (0, 255, 0)
            thickness = 3
        else:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 18), (x1 + w, y1), color, -1)
        cv2.putText(
            img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    os.makedirs("own_results", exist_ok=True)
    chosen_name = chosen_pred["name"] if chosen_pred else "NA"
    cv2.imwrite(f"own_results/all_bboxes_{chosen_name}_{rand}.jpg", img)


def predict_image(image, model, signal):
    """
    Predict the image using the model and save the results in the 'runs' folder

    Inputs
    ------
    image: str - name of the image file

    model: torch.hub.load - model to be used for prediction

    signal: str - signal to be used for filtering the predictions

    Returns
    -------
    str - predicted label
    """
    try:
        img_path = os.path.join("uploads", image)
        img = Image.open(img_path)

        # Predict the image using the model
        results = model.predict(
            source=img_path, save=True, conf=0.6, classes=[27, 28, 30]
        )
        result = results[0]

        # Save annotated image to runs folder
        os.makedirs("runs", exist_ok=True)
        annotated = result.plot()
        cv2.imwrite(os.path.join("runs", image), annotated)

        # Extract predictions from YOLOv8 results into a list of dicts
        boxes = result.boxes
        pred_list = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            xmin, ymin, xmax, ymax = (
                float(xyxy[0]),
                float(xyxy[1]),
                float(xyxy[2]),
                float(xyxy[3]),
            )
            conf = float(boxes.conf[i].cpu())
            cls_idx = int(boxes.cls[i].cpu())
            name = result.names[cls_idx]
            bbox_area = (xmax - xmin) * (ymax - ymin)
            pred_list.append(
                {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "confidence": conf,
                    "name": name,
                    "bboxArea": bbox_area,
                }
            )

        # Sort by bboxArea descending (largest first)
        pred_list.sort(key=lambda x: x["bboxArea"], reverse=True)

        # Filter out Bullseye/marker (model uses '10' for Bullseye, 'marker' for marker)
        # pred_list = [
        #     p for p in pred_list if p["name"] not in ("Bullseye", "10", "marker")
        # ]

        # Initialize prediction to NA
        pred = "NA"

        print(pred_list)

        if len(pred_list) == 1:
            pred = pred_list[0]

        # If more than 1 label is detected
        elif len(pred_list) > 1:
            # More than 1 Symbol detected, filter by confidence and area
            pred_shortlist = []
            current_area = pred_list[0]["bboxArea"]
            for row in pred_list:
                if row["confidence"] > 0.5 and (
                    (current_area * 0.8 <= row["bboxArea"])
                    or (
                        row["name"] in ("One", "11")
                        and current_area * 0.6 <= row["bboxArea"]
                    )
                ):
                    pred_shortlist.append(row)
                    current_area = row["bboxArea"]

            # If only 1 prediction remains after filtering
            if len(pred_shortlist) == 1:
                pred = pred_shortlist[0]

            # If multiple predictions remain, use signal to filter further
            elif len(pred_shortlist) > 1:
                pred_shortlist.sort(key=lambda x: x["xmin"])

                if signal == "L":
                    pred = pred_shortlist[0]
                elif signal == "R":
                    pred = pred_shortlist[-1]
                else:
                    # Signal is 'C', choose the prediction that is central in the image
                    for p in pred_shortlist:
                        if 250 < p["xmin"] < 774:
                            pred = p
                            break
                    # If no prediction is central, choose the one with the largest area
                    if isinstance(pred, str):
                        pred = max(pred_shortlist, key=lambda x: x["bboxArea"])

        # Save image with ALL detected bounding boxes to own_results
        os.makedirs("own_results", exist_ok=True)
        if pred_list:
            draw_all_bboxes(
                np.array(img), pred_list, pred if not isinstance(pred, str) else None
            )
        else:
            # No detections — save raw image so we can still see what was captured
            rand = str(int(time.time()))
            raw = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"own_results/all_bboxes_NA_{rand}.jpg", raw)

        # The model's class names are already image IDs (e.g. '15', '36'),
        # so we use them directly as the image_id
        if not isinstance(pred, str):
            image_id = str(pred["name"])
        else:
            image_id = "NA"
        print(f"Final result: {image_id}")
        return image_id
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Final result: NA (error: {e})")
        return "NA"


def predict_image_week_9(image, model):
    img_path = os.path.join("uploads", image)
    img = Image.open(img_path)

    # Run inference
    results = model(img_path)
    result = results[0]

    # Save annotated image to runs folder
    os.makedirs("runs", exist_ok=True)
    annotated = result.plot()
    cv2.imwrite(os.path.join("runs", image), annotated)

    # Extract predictions from YOLOv8 results
    boxes = result.boxes
    pred_list = []
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy()
        xmin, ymin, xmax, ymax = (
            float(xyxy[0]),
            float(xyxy[1]),
            float(xyxy[2]),
            float(xyxy[3]),
        )
        conf = float(boxes.conf[i].cpu())
        cls_idx = int(boxes.cls[i].cpu())
        name = result.names[cls_idx]
        bbox_area = (xmax - xmin) * (ymax - ymin)
        pred_list.append(
            {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "confidence": conf,
                "name": name,
                "bboxArea": bbox_area,
            }
        )

    # Sort by bboxArea descending (largest first)
    pred_list.sort(key=lambda x: x["bboxArea"], reverse=True)

    pred = "NA"
    # Go through the predictions, and choose the first one with confidence > 0.5
    for row in pred_list:
        if row["name"] not in ("Bullseye", "10", "marker") and row["confidence"] > 0.5:
            pred = row
            break

    # Save image with ALL detected bounding boxes to own_results
    os.makedirs("own_results", exist_ok=True)
    if pred_list:
        draw_all_bboxes(
            np.array(img), pred_list, pred if not isinstance(pred, str) else None
        )
    else:
        # No detections — save raw image so we can still see what was captured
        rand = str(int(time.time()))
        raw = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"own_results/all_bboxes_NA_{rand}.jpg", raw)

    # The model's class names are already image IDs
    if not isinstance(pred, str):
        image_id = str(pred["name"])
    else:
        image_id = "NA"
    return image_id


def stitch_image():
    """
    Stitches the images in the folder together and saves it into runs/stitched folder
    """
    # Initialize path to save stitched image
    imgFolder = "runs"
    stitchedPath = os.path.join(imgFolder, f"stitched-{int(time.time())}.jpeg")

    # Find all files that ends with ".jpg" (this won't match the stitched images as we name them ".jpeg")
    imgPaths = glob.glob(os.path.join(imgFolder + "/detect/*/", "*.jpg"))
    if not imgPaths:
        return None
    # Open all images
    images = [Image.open(x) for x in imgPaths]
    # Get the width and height of each image
    width, height = zip(*(i.size for i in images))
    # Calculate the total width and max height of the stitched image, as we are stitching horizontally
    total_width = sum(width)
    max_height = max(height)
    stitchedImg = Image.new("RGB", (total_width, max_height))
    x_offset = 0

    # Stitch the images together
    for im in images:
        stitchedImg.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    # Save the stitched image to the path
    stitchedImg.save(stitchedPath)

    # Move original images to "originals" subdirectory
    for img in imgPaths:
        shutil.move(img, os.path.join("runs", "originals", os.path.basename(img)))

    return stitchedImg


def stitch_image_own():
    """
    Stitches the annotated images in own_results into a grid collage and saves to stitched_output folder.
    Images are resized to 320x320 and arranged in columns of 2.
    """
    imgFolder = "own_results"
    outputFolder = "stitched_output"
    os.makedirs(outputFolder, exist_ok=True)

    imgPaths = glob.glob(os.path.join(imgFolder, "all_bboxes_*.jpg"))
    if not imgPaths:
        imgPaths = glob.glob(os.path.join(imgFolder, "annotated_image_*.jpg"))
    if not imgPaths:
        return None

    # Sort by timestamp (last part of filename before .jpg)
    imgTimestamps = [imgPath.split("_")[-1][:-4] for imgPath in imgPaths]
    sortedByTimeStamp = sorted(zip(imgPaths, imgTimestamps), key=lambda x: x[1])

    images = []
    for path, _ in sortedByTimeStamp:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (320, 320))
            images.append(img)

    if not images:
        return None

    # Arrange in columns of 2 rows each
    # Pad with blank images if odd number
    if len(images) % 2 != 0:
        images.append(np.zeros((320, 320, 3), dtype=np.uint8))

    columns = []
    for i in range(0, len(images), 2):
        col = np.vstack([images[i], images[i + 1]])
        columns.append(col)

    canvas = np.hstack(columns)

    stitchedPath = os.path.join(outputFolder, f"stitched-{int(time.time())}.jpg")
    cv2.imwrite(stitchedPath, canvas)

    return Image.open(stitchedPath)
