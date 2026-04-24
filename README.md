# Real-Time Indian Sign Language (ISL) Recognition - Prototype

This is a **working student demo prototype** for internship review.
It focuses on a clean, modular pipeline over production-grade accuracy.

---

## ✅ Features

- CNN model using TensorFlow/Keras for A-Z alphabet classes.
- Real-time webcam inference using OpenCV.
- Streamlit UI for live feed + prediction output.
- Confidence display for predictions.
- "No hand detected" handling (simple heuristic).
- Clear behavior when model is not loaded (no fake/random predictions).

---

## 📁 Project Structure

```bash
.
├── app/
│   └── main.py
├── model/
│   ├── train.py
│   └── model.h5                # generated after training
├── utils/
│   └── preprocessing.py
├── requirements.txt
└── README.md
```

---

## Requirement-to-Implementation Mapping

## Requirement IDs (Spec Trace)

- **REQ-1**: CNN model, preprocessing, training, and `.h5` save.
- **REQ-2**: OpenCV webcam frame capture + inference backend flow.
- **REQ-3**: Basic Streamlit UI with live feed + prediction output.
- **REQ-4**: Required folder/file structure (`model/`, `app/`, `utils/`).
- **REQ-5**: Real-time detection, confidence output, and no-hand handling.
- **REQ-6**: Clean modular prototype implementation.
- **REQ-7**: Colab-friendly training instructions.

---

1. **CNN model + training + preprocessing + .h5 save**
   - Implemented in `model/train.py` (`build_cnn_model`, `load_datasets`, `train_and_save`).
2. **OpenCV backend frame pipeline**
   - Implemented in `app/main.py` (`VideoCapture`, per-frame prediction loop).
3. **Basic frontend with live feed + real-time output**
   - Implemented in `app/main.py` using Streamlit.
4. **Requested modular structure**
   - Implemented exactly with `app/`, `model/`, `utils/` folders.
5. **Confidence + no-hand errors**
   - Confidence from softmax max probability; no-hand from heuristic in `utils/preprocessing.py`.

---

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2) Train model (Google Colab recommended)

Dataset (Kaggle):
https://www.kaggle.com/datasets/soumyakushwaha/indian-sign-language-dataset

### Colab workflow

1. Open Google Colab.
2. Upload this project or clone your repository.
3. Upload/download dataset and extract to a folder with class subfolders `A ... Z`.
4. Run training:

```bash
!python model/train.py --data-dir /content/indian-sign-language-dataset --epochs 10 --output model/model.h5
```

After training, download `model/model.h5` and place it in your project `model/` folder.

---

## 3) Run real-time app

```bash
streamlit run app/main.py
```

- Default model path in UI: `model/model.h5`
- If model cannot load, app clearly reports that prediction is unavailable.

---

## Notes

- Preprocessing in both training and inference: **resize to 64x64 + normalize**.
- No-hand detection uses a simple skin-color mask heuristic for demo only.
- This is a prototype intended for review/demo purposes.
