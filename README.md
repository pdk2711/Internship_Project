# Real-Time Indian Sign Language Recognition (Prototype)

A minimal working demo for student project review.

## What this project includes

- Real-time webcam UI using Streamlit + OpenCV.
- CNN model definition using TensorFlow/Keras.
- Training script for the Kaggle Indian Sign Language dataset.
- Safe fallback behavior: if model is missing, app uses random A-Z predictions.

---

## Project Structure

```bash
.
├── app.py             # Streamlit UI + webcam inference loop
├── model.py           # CNN architecture + model save/load helpers
├── train.py           # Dataset loading + training pipeline
├── utils.py           # Preprocessing + prediction + dummy fallback
├── requirements.txt   # Dependencies
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## 1) Download dataset (Kaggle)

Dataset: https://www.kaggle.com/datasets/soumyakushwaha/indian-sign-language-dataset

After downloading/unzipping, keep folder structure like:

```bash
data/
  A/
  B/
  ...
  Z/
```

---

## 2) Train the CNN

```bash
python train.py --data-dir data --epochs 8 --output-model isl_model.h5
```

This will train a small CNN and save `isl_model.h5`.

---

## 3) Run live demo

```bash
streamlit run app.py
```

In the sidebar, keep model path as `isl_model.h5` (or set your custom path).

---

## Notes

- This is a prototype for demo/review, not production-grade accuracy.
- If model file is not present, the app still runs with placeholder random predictions.
- Webcam permission is required for live feed.
