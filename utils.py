"""Utility functions for frame preprocessing and prediction."""

import random
from typing import Optional

import cv2
import numpy as np
from tensorflow.keras.models import Model


ALPHABETS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize frame to 64x64, normalize to [0,1], and add batch dimension."""
    resized = cv2.resize(frame, (64, 64))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)


def dummy_prediction() -> str:
    """Return a random alphabet class when no trained model is present."""
    return random.choice(ALPHABETS)


def predict_alphabet(input_tensor: np.ndarray, model: Optional[Model]) -> str:
    """Predict alphabet label from input tensor using model or fallback dummy output."""
    if model is None:
        return dummy_prediction()

    preds = model.predict(input_tensor, verbose=0)
    index = int(np.argmax(preds, axis=1)[0])
    return ALPHABETS[index]
