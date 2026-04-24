"""Training pipeline for ISL alphabet CNN (A-Z).

SPEC TRACE (Model):
- [REQ-1] CNN architecture using TensorFlow/Keras.
- [REQ-1] Dataset preprocessing (resize + normalize).
- [REQ-1] Save trained model as .h5 file.
- [REQ-7] Designed to run in Google Colab via CLI arguments.

Dataset should be organized as image folders by class name:

<dataset_root>/
  A/
  B/
  ...
  Z/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_CLASSES = 26


def build_cnn_model(input_shape: tuple[int, int, int] = (64, 64, 3)) -> tf.keras.Model:
    """Build and compile a compact CNN for alphabet classification.

    SPEC TRACE:
    - [REQ-1] CNN model definition with Conv + Pool + Dense.
    """
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_datasets(data_dir: Path, validation_split: float = 0.2):
    """Load train and validation datasets with resize + normalization.

    SPEC TRACE:
    - [REQ-1] Image resize to 64x64 and normalization to [0,1].
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        validation_split=validation_split,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        validation_split=validation_split,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    normalization = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def train_and_save(data_dir: str, epochs: int, output_path: str) -> None:
    """Train model and save .h5 output.

    SPEC TRACE:
    - [REQ-1] Train on A-Z dataset and save as .h5.
    """
    dataset_path = Path(data_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    train_ds, val_ds = load_datasets(dataset_path)
    model = build_cnn_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        )
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_file)
    print(f"✅ Saved trained model: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ISL CNN model")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to dataset folder containing A-Z subfolders",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--output",
        default="model/model.h5",
        help="Path to save trained .h5 model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save(args.data_dir, args.epochs, args.output)
