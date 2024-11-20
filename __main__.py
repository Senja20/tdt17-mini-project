from ultralytics import YOLO

from utils import get_device

device = get_device()

model = YOLO("yolov9c.pt")

model.info()

# essential training parameters
train_params = {
    "data": "./data.yaml",
    "imgsz": (1024, 128),  # (width, height)
    "plots": True,  # plot results.txt as results.png
    "workers": 8,  # workers are used for loading data
    "batch": 16,
    "rect": True,  # rectangular training (keep aspect ratio)
    "patience": 50,  # early stopping patience
    "project": "pole_detection",  # save to project/name
}

# Hyperparameters
hyper_params = {
    "epochs": 100,
    "name": "model_optimization",
    "optimizer": "AdamW",  # Optimize for mobile
    "lr0": 0.01,
    "lrf": 0.0001,
    "momentum": 0.937,
}

train_params = {**train_params, **hyper_params}

results = model.train(**train_params)

print(results)

model.save("yolov9c_trained.pt")

print("Done training")
