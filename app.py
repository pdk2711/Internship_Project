"""Streamlit app for a minimal Real-Time Indian Sign Language demo.

This prototype focuses on UI + inference flow for student project review.
"""

import cv2
import streamlit as st

from model import load_trained_model
from utils import ALPHABETS, predict_alphabet, preprocess_frame


st.set_page_config(page_title="ISL Recognition Demo", layout="wide")
st.title("🖐️ Real-Time Indian Sign Language Recognition (Prototype)")
st.caption("Minimal demo: webcam → preprocess → model/dummy prediction → UI")

# Sidebar: model configuration
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Model path (.h5)", value="isl_model.h5")

# Attempt model loading once and cache result for performance.
@st.cache_resource
def get_model(path: str):
    return load_trained_model(path)


model = get_model(model_path)
if model is None:
    st.warning(
        "No trained model found/loaded. Using random placeholder predictions (A–Z)."
    )
else:
    st.success("Trained model loaded successfully.")

# Webcam controls
col1, col2 = st.columns([1, 1])
with col1:
    start = st.button("Start Webcam")
with col2:
    stop = st.button("Stop Webcam")

frame_placeholder = st.empty()
pred_placeholder = st.empty()

# Persist run state in session state.
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

if start:
    st.session_state.run_camera = True
if stop:
    st.session_state.run_camera = False


# Main loop while camera is enabled.
if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Unable to access webcam. Please check camera permissions/device.")
        st.session_state.run_camera = False
    else:
        while st.session_state.run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break

            # Preprocess frame for model input: 64x64 + normalize.
            input_tensor = preprocess_frame(frame)

            # Predict A-Z using trained model, else fallback random prediction.
            predicted_label = predict_alphabet(input_tensor, model)

            # Convert BGR (OpenCV) to RGB for Streamlit display.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", caption="Live Webcam Feed")
            pred_placeholder.markdown(
                f"### Predicted Alphabet: **{predicted_label}**"
                + ("" if model else " (placeholder)")
            )

            # Small wait so Streamlit can update UI without freezing.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                st.session_state.run_camera = False
                break

        cap.release()
        cv2.destroyAllWindows()
else:
    # Idle state instructions.
    st.info("Click **Start Webcam** to begin live prediction.")
    pred_placeholder.markdown("### Predicted Alphabet: **-**")
    st.write("Supported output classes:", ", ".join(ALPHABETS))
