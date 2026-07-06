"""
Phase 4 - Exploratory Data Analysis (EDA)
Visualizes sample images, class distribution, and basic dataset statistics.
Saves plots to the results/ folder.
"""

import os
import random
import matplotlib.pyplot as plt
from PIL import Image

DATASET_DIR = "../dataset"
RESULTS_DIR = "../results"
CLASSES = ["WithMask", "WithoutMask"]


def count_images(split):
    counts = {}
    for cls in CLASSES:
        cls_path = os.path.join(DATASET_DIR, split, cls)
        if os.path.exists(cls_path):
            counts[cls] = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
        else:
            counts[cls] = 0
    return counts


def plot_class_distribution():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    splits = ["train", "validation", "test"]
    fig, axes = plt.subplots(1, len(splits), figsize=(15, 4))

    for ax, split in zip(axes, splits):
        counts = count_images(split)
        ax.bar(counts.keys(), counts.values(), color=["#2ecc71", "#e74c3c"])
        ax.set_title(f"{split.capitalize()} Set Class Distribution")
        ax.set_ylabel("Number of Images")
        for i, v in enumerate(counts.values()):
            ax.text(i, v + 5, str(v), ha="center")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "class_distribution.png")
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


def plot_sample_images(split="train", n_per_class=4):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(len(CLASSES), n_per_class, figsize=(n_per_class * 2.5, 5))

    for row, cls in enumerate(CLASSES):
        cls_path = os.path.join(DATASET_DIR, split, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        sample = random.sample(images, min(n_per_class, len(images)))

        for col, img_name in enumerate(sample):
            img = Image.open(os.path.join(cls_path, img_name))
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(cls, fontsize=12)
        axes[row, 0].set_title(cls, loc="left", fontsize=12, fontweight="bold")

    plt.suptitle(f"Sample Images from {split.capitalize()} Set")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "sample_images.png")
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


def print_statistics():
    print("=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    total = 0
    for split in ["train", "validation", "test"]:
        counts = count_images(split)
        split_total = sum(counts.values())
        total += split_total
        print(f"\n{split.upper()} ({split_total} images):")
        for cls, count in counts.items():
            pct = (count / split_total * 100) if split_total else 0
            print(f"  {cls}: {count} ({pct:.1f}%)")
    print(f"\nTOTAL IMAGES: {total}")


if __name__ == "__main__":
    print_statistics()
    plot_class_distribution()
    plot_sample_images()
