# Real-Time Indian Sign Language Recognition (Prototype)

A **minimal working demo** for student project review.

This project demonstrates a simple pipeline:

1. Capture webcam frame using OpenCV
2. Preprocess frame (`64x64`, normalized)
3. Run prediction using a loaded TensorFlow/Keras `.h5` model (if available)
4. Fallback to random alphabet prediction (`A-Z`) if no trained model exists
5. Display live feed + predicted alphabet in Streamlit UI

---

## Project Structure

```bash
.
├── app.py             # Streamlit UI + webcam inference loop
├── model.py           # CNN architecture + model save/load helpers
├── utils.py           # Preprocessing + prediction + dummy fallback
├── requirements.txt   # Dependencies
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

(Optional) create and save an untrained model structure:

```bash
python model.py
```

This saves `isl_model_structure.h5`.

---

## Run

```bash
streamlit run app.py
```

---

## Notes

- This is **not a production model**.
- The CNN is defined but not trained in this prototype.
- If a valid trained `.h5` is not found, the app still runs and shows placeholder random predictions.
- For demo use, provide webcam permission to your system/browser.
