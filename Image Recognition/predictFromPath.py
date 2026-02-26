from ultralytics import YOLO
import cv2
import os

model = YOLO("Image Recognition/runs/seg_v4.pt")

imgpath1 = os.path.relpath("C:/Users/Caleb/Documents/MDP Image Recognition/Data/rawImages/library_stony_3.jpg")
imgpath2 = os.path.relpath("C:/Users/Caleb/Documents/MDP Image Recognition/Data/rawImages/photo_2026-02-25_16-01-55.jpg")
imgpath3 = os.path.relpath("C:/Users/Caleb/Documents/MDP Image Recognition/Data/MDP.v7i/test/images/IMG_2480_jpg.rf.13c46008317bb802229122f8d9f7dafc.jpg")
imgpath4 = os.path.relpath("C:/Users/Caleb/Documents/MDP Image Recognition/Data/MDP.v7i/test/images/IMG_2371_jpg.rf.868ead450d8a03304e7389f6eed837a8.jpg")

def predict():
    results = model.predict(source=[imgpath1, imgpath2, imgpath3, imgpath4], show = True)
    for r in results:
        clss_list = r.boxes.cls.int().tolist()
        for cls in clss_list:
            print(f"Class name: {model.names[cls]}") # model.names[]

predict()
#print(model.names) # This shows the dictionary of internal class index to imageID
#r.boxes.cls.int().tolist() gets the internal class index so by feeding this into model.names, we can get the image id