from ultralytics import YOLO

from utils import get_device

device = get_device()

model = YOLO("yolov9c.pt")

model.info()

train_params = {
    "data": "./data.yaml",
    "epochs": 100,
    "imgsz": (1024, 128),  # (width, height)
    "plots": True,  # plot results.txt as results.png
    "workers": 4,  # workers are used for loading data
    "batch": 16,
    "rect": True,  # rectangular training (keep aspect ratio)
    "patience": 50,  # early stopping patience
}

results = model.train(**train_params)

print(results)

model.save("yolov9c_trained.pt")

print("Done training")
