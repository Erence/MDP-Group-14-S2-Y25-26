from ultralytics import YOLO
import cv2
import os

model = YOLO("Image Recognition/runs/task2_v4.pt")
#model = YOLO("runs/segment/task2_v4/weights/best.pt")

imgpath1 = os.path.relpath("C:/Users/Caleb/Downloads/Telegram Desktop/photo_2026-03-20_00-27-30.jpg")
imgpath2 = os.path.relpath("C:/Users/Caleb/Documents/MDP Image Recognition/Data/Task2/train/images/IMG_3287_jpg.rf.b4dc2be0b8aea3feb28d6bf56cfd5f9d.jpg")
imgpath3 = os.path.relpath("C:/Users/Caleb/Documents/MDP Image Recognition/Data/Task2/train/images/IMG_2709_jpg.rf.36ea69be18c10791a3d2c98827e307e8.jpg")
imgpath4 = os.path.relpath("C:/Users/Caleb/Documents/MDP Image Recognition/Data/Task2/test/images/right14_jpeg_jpg.rf.51f03ed916f3857f319c152cf436e22d.jpg")
imgpath5 = os.path.relpath("C:/Users/Caleb/Documents/MDP Image Recognition/Data/Task2/test/images/rightNight4_jpeg_jpg.rf.8b507b952ef53f8c7536f214406ed94f.jpg")
imgpath6 = os.path.relpath("C:/Users/Caleb/Documents/MDP Image Recognition/Data/Task2/test/images/left19_jpeg_jpg.rf.56c2d4b3c922ab037153a42d7bb206f7.jpg")
imgpath7 = os.path.relpath("C:/Users/Caleb/Documents/MDP Image Recognition/Data/Task2/test/images/left25_jpeg_jpg.rf.1eb9a6a435b52a92babaf97937c07501.jpg")
imgpath8 = os.path.relpath("C:/Users/Caleb/Documents/MDP Image Recognition/Data/Task2/train/images/IMG_2384_jpg.rf.2ee1200a8f92bed5508e572b45942a87.jpg")

def predict():
    results = model.predict(source=[imgpath1], 
                            show = True
                            #save = True
                            #classes = [27,28,30]
                            )
    for r in results:
        clss_list = r.boxes.cls.int().tolist()
        for cls in clss_list:
            print(f"Class name: {model.names[cls]}") # model.names[]

predict()
#print(model.names) # This shows the dictionary of internal class index to imageID
#r.boxes.cls.int().tolist() gets the internal class index so by feeding this into model.names, we can get the image id