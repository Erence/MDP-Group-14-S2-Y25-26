# Install torch separately for your available CUDA version and ultralytics after
#pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128 # In this case, CUDA 12.8
#pip install ultralytics

from ultralytics import YOLO
from multiprocessing import freeze_support
import os
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("yolo26n-seg.pt") # Latest nano instance segmentation model
new_path = os.path.relpath("C:/Users/Caleb/Documents/MDP Image Recognition/Data/YuktoSC/data.yaml")

def main():
    results = model.train(data=new_path, 
                          name="seg_v2", 
                          epochs=50, 
                          imgsz=640, 
                          device=device,
                          patience=25,
                          flipud=0.0, 
                          fliplr=0.0)
    results = model.val()

if __name__ == '__main__':
    freeze_support()
    main()
