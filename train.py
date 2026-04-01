"""Train CNN on Indian Sign Language Kaggle dataset.

Expected dataset layout after download/unzip:

<dataset_dir>/
  A/
  B/
  ...
  Z/

Usage:
    python train.py --data-dir /path/to/dataset --epochs 8 --output-model isl_model.h5
"""

import argparse
from pathlib import Path

import tensorflow as tf

from model import build_cnn_model


IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32


def build_datasets(data_dir: Path, validation_split: float = 0.2):
    """Create train/validation datasets from class-subfolder image directory."""
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

    # Cache + prefetch for faster training.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    return train_ds, val_ds


def train(data_dir: str, epochs: int, output_model: str) -> None:
    """Train model and save final .h5 file."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    train_ds, val_ds = build_datasets(data_path)

    model = build_cnn_model()

    # Add simple rescaling as the first preprocessing step for all images.
    normalization = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save(output_model)
    print(f"Saved trained model to: {output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISL CNN model")
    parser.add_argument("--data-dir", required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument(
        "--output-model",
        default="isl_model.h5",
        help="Output model path (.h5)",
    )

    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.output_model)
