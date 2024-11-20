from ultralytics import YOLO

from utils import get_device

# Device configuration
device = get_device()

# Load the YOLO model
model = YOLO("yolov9s.pt")
model.info()

# Define training parameters
train_params = {
    "data": "./data.yaml",  # Path to your dataset configuration
    "imgsz": (1024, 128),  # Image size (width, height)
    "plots": True,  # Enable result plotting
    "workers": 8,  # Number of data loading workers
    "batch": 16,  # Batch size
    "rect": True,  # Rectangular training (maintain aspect ratio)
    "patience": 10,  # Early stopping patience
    "project": "pole_detection",  # Project name for saving results
}

# Define initial hyperparameters
hyper_params = {
    "epochs": 50,  # Number of training epochs
    "optimizer": "AdamW",  # Optimizer choice
    "lr0": 0.01,  # Initial learning rate
    "lrf": 0.0001,  # Final learning rate
    "momentum": 0.937,  # Momentum
    "warmup_epochs": 5,  # Number of warmup epochs
    "dropout": 0.1,  # Dropout rate
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
