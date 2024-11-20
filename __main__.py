import optuna
from ultralytics import YOLO

from utils import get_device

# Base training parameters (shared by all runs)
BASE_TRAIN_PARAMS = {
    "data": "./data.yaml",  # Path to your dataset config
    "imgsz": (1024, 128),  # Input image size (width, height)
    "plots": False,  # Disable plots for faster trials
    "workers": 8,  # Number of dataloader workers
    "rect": True,  # Rectangular training
    "patience": 20,  # Early stopping patience
    "optimizer": "AdamW",  # Optimizer type
}

# Augmentation parameters (shared defaults)
BASE_AUGMENT_PARAMS = {
    "augment": True,  # Enable augmentations
}


# Objective function for Optuna
def objective(trial):
    device = get_device()

    # Sample hyperparameters
    hyper_params = {
        "lr0": trial.suggest_float(
            "lr0", 1e-5, 1e-1, log=True
        ),  # Initial learning rate
        "lrf": trial.suggest_float(
            "lrf", 1e-5, 1e-2, log=True
        ),  # Final learning rate fraction
        "momentum": trial.suggest_float("momentum", 0.8, 0.99),  # Momentum
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),  # Dropout rate
        "epochs": trial.suggest_int("epochs", 50, 300, step=50),  # Number of epochs
        "batch": trial.suggest_int("batch", 8, 64, step=8),  # Batch size
    }

    # Sample augmentation parameters
    augment_params = {
        "translate": trial.suggest_float("translate", 0.0, 0.2),  # Translation
        "scale": trial.suggest_float("scale", 0.1, 1.0),  # Scaling
        "flipud": trial.suggest_float("flipud", 0.0, 0.5),  # Vertical flip probability
        "fliplr": trial.suggest_float("fliplr", 0.0, 0.5),
    }

    # Combine all parameters
    params = {
        **BASE_TRAIN_PARAMS,
        **hyper_params,
        **BASE_AUGMENT_PARAMS,
        **augment_params,
    }

    # Initialize and train the YOLO model
    model = YOLO("yolov9c.pt")  # Path to your YOLO model
    results = model.train(**params)

    # Return validation loss as the objective metric
    val_loss = results.box_loss  # Use `box_loss` as an example metric
    return val_loss


# Run Optuna optimization
def run_optimization():
    # Create a study for minimizing validation loss
    study = optuna.create_study(direction="minimize")

    # Optimize the objective function
    study.optimize(
        objective, n_trials=20, timeout=3600
    )  # Run 20 trials or stop after 1 hour

    # Print the best hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study


# Train final model with the best hyperparameters
def train_final_model(study):
    # Extract best hyperparameters
    best_hyper_params = study.best_params

    # Final training parameters
    final_params = {**BASE_TRAIN_PARAMS, **BASE_AUGMENT_PARAMS, **best_hyper_params}

    # Enable plots for final training
    final_params["plots"] = True
    final_params["name"] = "best_model"  # Name for the final model
    final_params["project"] = "pole_detection_final"

    # Initialize and train the YOLO model
    model = YOLO("yolov9c.pt")  # Path to your YOLO model
    results = model.train(**final_params)

    # Save the trained model
    model.save("yolov9c_trained_optimized.pt")
    print("Final model training completed and saved as 'yolov9c_trained_optimized.pt'")


if __name__ == "__main__":
    # Run Optuna optimization
    study = run_optimization()

    # Train the final model with the best hyperparameters
    train_final_model(study)
