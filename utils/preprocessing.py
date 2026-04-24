"""Preprocessing helpers for real-time ISL recognition.

SPEC TRACE (Utilities):
- [REQ-1/REQ-2] Shared preprocessing for training/inference.
- [REQ-5] Basic no-hand-detected heuristic support.
"""

from __future__ import annotations

import cv2
import numpy as np

# Model input size
IMAGE_SIZE = (64, 64)

# Ordered class labels for A-Z
ALPHABETS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize + normalize frame and return model-ready batch tensor.

    SPEC TRACE:
    - [REQ-1/REQ-2] Resize to 64x64 and normalize before prediction/training.


    Args:
        frame: BGR image from OpenCV webcam.

    Returns:
        Tensor with shape (1, 64, 64, 3), values in [0, 1].
    """
    resized = cv2.resize(frame, IMAGE_SIZE)
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)


def detect_hand_presence(frame: np.ndarray, min_area: int = 3000) -> bool:
    """Very simple hand-presence heuristic using skin-color mask area.

    This is a lightweight demo heuristic (not robust in all lighting conditions).
    It helps provide a friendly 'No hand detected' state instead of always predicting.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Approximate skin range in HSV (demo-friendly, may need tuning).
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 180, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean noisy mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    area = int(cv2.countNonZero(mask))
    return area >= min_area


def decode_prediction(predictions: np.ndarray) -> tuple[str, float]:
    """Convert model softmax output to alphabet and confidence."""
    idx = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions, axis=1)[0])
    return ALPHABETS[idx], confidence
