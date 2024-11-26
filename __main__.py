from ultralytics import YOLO

from utils_functions import get_device

# Device configuration
device = get_device()

print("Device:", device)

# Load the YOLO model
model = YOLO("yolo11s.yaml")
model.info()

train_params = {
    "data": "./data.yaml",
    # "imgsz": (1024, 128),
    "imgsz": 800,
    "plots": True,
    "workers": 4,
    "batch": 16,
    # "rect": True,
    "patience": 10,
    "project": "pole_detection",
}

hyper_params = {
    "weight_decay": 0.0005,
    "epochs": 100,
    "optimizer": "AdamW",
    "lr0": 0.01117,
    "momentum": 0.86731,
    "warmup_epochs": 5.0,
    "lrf": 0.0001,
    "dropout": 0.1,
}

# Define augmentation parameters
augmentation_params = {
    "augment": True,  # Enable data augmentation
    "hsv_h": 0.01702,  # HSV-Hue augmentation
    "hsv_s": 0.7131,  # HSV-Saturation augmentation
    "hsv_v": 0.42673,  # HSV-Value augmentation
    "degrees": 0.0,  # Rotation augmentation
    "translate": 0.09982,  # Translation augmentation
    "scale": 0.50072,  # Scale augmentation
    "shear": 0.0,  # Shear augmentation
    "perspective": 0.00,  # Perspective augmentation
    "flipud": 0.0,  # Vertical flip augmentation
    "fliplr": 0.35743,  # Horizontal flip augmentation
    "mosaic": 0.90831,  # Mosaic augmentation
    "mixup": 0.0,  # Mixup augmentation
    "copy_paste": 0.0,  # Copy-paste augmentation
    "erasing": 0.0,  # Random erasing augmentation
    "crop_fraction": 0.0,  # Crop fraction
}

# Combine all parameters
train_params.update(hyper_params)
train_params.update(augmentation_params)

# Perform hyperparameter tuning
# https://docs.ultralytics.com/guides/hyperparameter-tuning/#repeat
results = model.train(
    **train_params,
    save=True,  # Disable saving intermediate models
    val=True,  # Disable validation during tuning
    device=device,
)
print("Results: ", results)
# Save the trained model
model_path = model.export(
    format="onnx",
    device="cpu",
    imgsz=864,
)

print(f"Model saved to {model_path}")
print("Training completed successfully.")
