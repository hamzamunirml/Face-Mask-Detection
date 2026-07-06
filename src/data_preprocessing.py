"""
Phase 3 - Data Preprocessing
Handles image resizing, normalization, label encoding, and data augmentation
using Keras ImageDataGenerator.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


def get_data_generators(dataset_dir="../dataset"):
    """
    Returns train, validation and test generators with:
    - Pixel normalization (rescale 1./255)
    - Data augmentation on the training set (rotation, zoom, flip, shift)
    - No augmentation on validation/test (only rescale)
    """

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        f"{dataset_dir}/train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )

    validation_generator = val_test_datagen.flow_from_directory(
        f"{dataset_dir}/validation",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    test_generator = val_test_datagen.flow_from_directory(
        f"{dataset_dir}/test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    print("Class indices:", train_generator.class_indices)

    return train_generator, validation_generator, test_generator


if __name__ == "__main__":
    train_gen, val_gen, test_gen = get_data_generators()
    print(f"Train samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
