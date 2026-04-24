"""Streamlit app for real-time Indian Sign Language alphabet prediction.

SPEC TRACE (Frontend + Backend):
- [REQ-2] OpenCV webcam capture and per-frame processing.
- [REQ-3] Streamlit UI showing webcam feed + real-time output.
- [REQ-5] Confidence display and no-hand-detected handling.
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import streamlit as st
from tensorflow.keras.models import load_model

# Allow importing from sibling utils directory when run via streamlit.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.preprocessing import decode_prediction, detect_hand_presence, preprocess_frame


@st.cache_resource
def get_model(model_path: str):
    """Load trained model once; return None if not found/invalid."""
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        return load_model(path)
    except Exception:
        return None


def predict_frame(frame, model):
    """Predict alphabet + confidence from a single webcam frame.

    SPEC TRACE:
    - [REQ-2] Preprocess each frame before prediction.
    - [REQ-5] Return explicit error state when no hand is detected.
    """
    if model is None:
        return "Model not loaded", 0.0

    if not detect_hand_presence(frame):
        return "No hand detected", 0.0

    input_tensor = preprocess_frame(frame)
    predictions = model.predict(input_tensor, verbose=0)
    label, confidence = decode_prediction(predictions)
    return label, confidence


def main():
    st.set_page_config(page_title="Real-Time ISL Recognition", layout="wide")
    st.title("🖐️ Real-Time Indian Sign Language (ISL) Recognition")
    st.caption("Prototype demo: webcam + CNN inference + confidence output")

    st.sidebar.header("Settings")
    default_model_path = str(ROOT_DIR / "model" / "model.h5")
    model_path = st.sidebar.text_input("Model path (.h5)", value=default_model_path)
    model = get_model(model_path)

    if model is None:
        st.warning("Model not loaded. Please train or provide a valid .h5 model path.")
    else:
        st.success("Model loaded successfully.")

    if "run_camera" not in st.session_state:
        st.session_state.run_camera = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Camera"):
            st.session_state.run_camera = True
    with col2:
        if st.button("Stop Camera"):
            st.session_state.run_camera = False

    frame_slot = st.empty()
    pred_slot = st.empty()

    if st.session_state.run_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam. Please check webcam permissions.")
            st.session_state.run_camera = False
            return

        while st.session_state.run_camera:
            ok, frame = cap.read()
            if not ok:
                st.error("Failed to read webcam frame.")
                break

            label, confidence = predict_frame(frame, model)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_slot.image(rgb, channels="RGB", caption="Live Webcam Feed")

            if label == "Model not loaded":
                pred_slot.warning("Model not loaded. Prediction is unavailable.")
            elif label == "No hand detected":
                pred_slot.info("✋ No hand detected. Please place hand in front of camera.")
            else:
                pred_slot.markdown(
                    f"### Prediction: **{label}**  \nConfidence: **{confidence:.2%}**"
                )

        cap.release()
        cv2.destroyAllWindows()
    else:
        st.info("Click **Start Camera** to begin real-time prediction.")


if __name__ == "__main__":
    main()
