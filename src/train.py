"""
Phase 5 - Model Development (Training)
Trains all three models (Custom CNN, MobileNetV2, ResNet50) and saves
the best checkpoint of each to the models/ folder, along with training
history plots to results/.

Usage:
    python train.py --model all
    python train.py --model custom_cnn
    python train.py --model mobilenetv2
    python train.py --model resnet50
"""

import os
import argparse
import json
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_preprocessing import get_data_generators
from models import MODEL_BUILDERS

MODELS_DIR = "../models"
RESULTS_DIR = "../results"
EPOCHS = 25


def get_callbacks(model_name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f"{model_name}_best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]


def plot_history(history, model_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title(f"{model_name} - Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title(f"{model_name} - Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{model_name}_training_history.png")
    plt.savefig(out_path)
    print(f"Saved plot: {out_path}")

    # Save raw history as JSON too
    hist_path = os.path.join(RESULTS_DIR, f"{model_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history.history, f, indent=2)


def train_model(model_name):
    print(f"\n{'='*60}\nTraining: {model_name}\n{'='*60}")

    train_gen, val_gen, _ = get_data_generators()
    model = MODEL_BUILDERS[model_name]()
    model.summary()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=get_callbacks(model_name),
    )

    # Save final model too (SavedModel format)
    final_path = os.path.join(MODELS_DIR, f"{model_name}_final.h5")
    model.save(final_path)
    print(f"Saved final model: {final_path}")

    plot_history(history, model_name)
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["all", "custom_cnn", "mobilenetv2", "resnet50"],
        default="all",
    )
    args = parser.parse_args()

    if args.model == "all":
        for name in MODEL_BUILDERS:
            train_model(name)
    else:
        train_model(args.model)
