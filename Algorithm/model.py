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


def draw_own_bbox(
    img, x1, y1, x2, y2, label, color=(36, 255, 12), text_color=(0, 0, 0)
):
    """
    Draw bounding box on the image with text label and save both the raw and annotated image in the 'own_results' folder

    Inputs
    ------
    img: numpy.ndarray - image on which the bounding box is to be drawn

    x1: int - x coordinate of the top left corner of the bounding box

    y1: int - y coordinate of the top left corner of the bounding box

    x2: int - x coordinate of the bottom right corner of the bounding box

    y2: int - y coordinate of the bottom right corner of the bounding box

    label: str - label to be written on the bounding box

    color: tuple - color of the bounding box

    text_color: tuple - color of the text label

    Returns
    -------
    None

    """
    # Label is already the image ID from the model (e.g. '15', '36')
    label = str(label)
    # Convert the coordinates to int
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    # Create a random string to be used as the suffix for the image name, just in case the same name is accidentally used
    rand = str(int(time.time()))

    # Save the raw image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"own_results/raw_image_{label}_{rand}.jpg", img)

    # Draw the bounding box
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # For the text background, find space required by the text so that we can put a background with that amount of width.
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    # Print the text
    img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    img = cv2.putText(
        img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1
    )
    # Save the annotated image
    cv2.imwrite(f"own_results/annotated_image_{label}_{rand}.jpg", img)


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
        results = model.predict(source=img_path)
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

        # Draw the bounding box on the image
        if not isinstance(pred, str):
            draw_own_bbox(
                np.array(img),
                pred["xmin"],
                pred["ymin"],
                pred["xmax"],
                pred["ymax"],
                pred["name"],
            )

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

    # Draw the bounding box on the image
    if not isinstance(pred, str):
        draw_own_bbox(
            np.array(img),
            pred["xmin"],
            pred["ymin"],
            pred["xmax"],
            pred["ymax"],
            pred["name"],
        )

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
    Stitches the images in the folder together and saves it into own_results folder

    Basically similar to stitch_image() but with different folder names and slightly different drawing of bounding boxes and text
    """
    imgFolder = "own_results"
    stitchedPath = os.path.join(imgFolder, f"stitched-{int(time.time())}.jpeg")

    imgPaths = glob.glob(os.path.join(imgFolder + "/annotated_image_*.jpg"))
    imgTimestamps = [imgPath.split("_")[-1][:-4] for imgPath in imgPaths]

    sortedByTimeStampImages = sorted(zip(imgPaths, imgTimestamps), key=lambda x: x[1])

    images = [Image.open(x[0]) for x in sortedByTimeStampImages]
    width, height = zip(*(i.size for i in images))
    total_width = sum(width)
    max_height = max(height)
    stitchedImg = Image.new("RGB", (total_width, max_height))
    x_offset = 0

    for im in images:
        stitchedImg.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    stitchedImg.save(stitchedPath)

    return stitchedImg
