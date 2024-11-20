from ultralytics import YOLO

from utils import get_device

# Device configuration
device = get_device()

# Load the YOLO model
model = YOLO("yolov9s.pt")
model.info()

train_params = {
    "data": "./data.yaml",
    "imgsz": (1024, 128),
    "plots": True,
    "workers": 4,
    "batch": 16,
    "rect": True,
    "patience": 10,
    "project": "pole_detection",
}

hyper_params = {
    "epochs": 50,
    "optimizer": "AdamW",
    "lr0": 0.01,
    "lrf": 0.0001,
    "momentum": 0.937,
    "warmup_epochs": 5,
    "dropout": 0.1,
}

# Define augmentation parameters
augmentation_params = {
    "augment": True,  # Enable data augmentation
    "hsv_h": 0.015,  # HSV-Hue augmentation
    "hsv_s": 0.7,  # HSV-Saturation augmentation
    "hsv_v": 0.4,  # HSV-Value augmentation
    "degrees": 0.0,  # Rotation augmentation
    "translate": 0.1,  # Translation augmentation
    "scale": 0.5,  # Scale augmentation
    "shear": 0.0,  # Shear augmentation
    "perspective": 0.0,  # Perspective augmentation
    "flipud": 0.0,  # Vertical flip augmentation
    "fliplr": 0.5,  # Horizontal flip augmentation
    "mosaic": 1.0,  # Mosaic augmentation
    "mixup": 0.0,  # Mixup augmentation
    "copy_paste": 0.0,  # Copy-paste augmentation
    "erasing": 0.0,  # Random erasing augmentation
    "crop_fraction": 0.5,  # Crop fraction
}

# Combine all parameters
train_params.update(hyper_params)
train_params.update(augmentation_params)

# Perform hyperparameter tuning
# https://docs.ultralytics.com/guides/hyperparameter-tuning/#repeat
tuning_results = model.tune(
    **train_params,
    iterations=300,  # Number of tuning iterations
    save=True,  # Disable saving intermediate models
    val=False,  # Disable validation during tuning
)

# Train the model with the best hyperparameters found
best_hyperparams = tuning_results["best_hyperparameters"]
train_params.update(best_hyperparams)

train_params["epochs"] = 100  # Increase the number of epochs
train_params["val"] = True

# Perform training
results = model.train(**train_params)

# Save the trained model
model.save("yolo_trained.pt")

print("Training completed successfully.")
