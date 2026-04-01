"""Model utilities for ISL alphabet recognition prototype.

Includes:
- CNN architecture creation (untrained)
- Model structure saving
- Trained model loading from .h5
"""

from pathlib import Path
from typing import Optional

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Model, load_model


NUM_CLASSES = 26
INPUT_SHAPE = (64, 64, 3)


def build_cnn_model() -> Model:
    """Create a simple CNN model architecture for 26 alphabet classes."""
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def save_model_structure(output_path: str = "isl_model_structure.h5") -> str:
    """Save untrained model structure+initialized weights as an .h5 file.

    This is useful as a placeholder artifact for demo setup.
    """
    model = build_cnn_model()
    model.save(output_path)
    return output_path


def load_trained_model(model_path: str) -> Optional[Model]:
    """Load a trained .h5 model if available; return None on failure."""
    path = Path(model_path)
    if not path.exists() or path.suffix.lower() != ".h5":
        return None

    try:
        return load_model(path)
    except Exception:
        # Keep the app resilient for demo usage when model is missing/corrupt.
        return None


if __name__ == "__main__":
    saved = save_model_structure()
    print(f"Saved model structure to: {saved}")
