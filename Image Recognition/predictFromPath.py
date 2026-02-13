from ultralytics import YOLO
import cv2
import os

model = YOLO("Image Recognition/runs/seg_v2.pt")

imgpath1 = os.path.relpath("C:/Users/caleb/Documents/MDP Image Recognition/Data/YuktoSC/test/images/20240126_091022_jpg.rf.8b922c5702bb61149742583d22e84583.jpg")
imgpath2 = os.path.relpath("C:/Users/caleb/Documents/MDP Image Recognition/Data/YuktoSC/test/images/20240211_232039_jpg.rf.8ffbadc50a357c89d63a377e69e430ac.jpg")

def predict():
    results = model.predict(source=[imgpath1, imgpath2], show = True)
    for r in results:
        clss_list = r.boxes.cls.int().tolist()
        for cls in clss_list:
            print(f"Class name: {model.names[cls]}") # model.names[]

predict()
#print(model.names) # This shows the dictionary of internal class index to imageID
#r.boxes.cls.int().tolist() gets the internal class index so by feeding this into model.names, we can get the image id