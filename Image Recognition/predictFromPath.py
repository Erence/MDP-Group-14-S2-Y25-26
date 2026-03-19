from ultralytics import YOLO
import cv2
import os

model = YOLO("Image Recognition/runs/task2_v1.pt")
# model = YOLO("runs/segment/task2_v1/weights/last.pt")

imgpath1 = os.path.relpath("C:/Users/Caleb/Downloads/Telegram Desktop/photo_2026-03-20_00-23-29.jpg")
imgpath2 = os.path.relpath("C:/Users/Caleb/Downloads/Telegram Desktop/photo_2026-03-20_00-27-28.jpg")
imgpath3 = os.path.relpath("C:/Users/Caleb/Downloads/Telegram Desktop/photo_2026-03-20_00-27-30.jpg")
imgpath4 = os.path.relpath("C:/Users/Caleb/Documents/MDP Image Recognition/Data/MDP.v7i/train/images/IMG_2717_jpg.rf.35f0fd4704c5ad1b64a26b2baacb8917.jpg")

def predict():
    results = model.predict(source=[imgpath1, imgpath2, imgpath3], 
                            show = True,
                            save = True
                            #classes = [27,28,30]
                            )
    for r in results:
        clss_list = r.boxes.cls.int().tolist()
        for cls in clss_list:
            print(f"Class name: {model.names[cls]}") # model.names[]

predict()
#print(model.names) # This shows the dictionary of internal class index to imageID
#r.boxes.cls.int().tolist() gets the internal class index so by feeding this into model.names, we can get the image id