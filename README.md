# 😷 Real-Time Face Mask Detection Using Deep Learning and Computer Vision

An end-to-end deep learning system that classifies whether a person is wearing a face mask, with support for image uploads, real-time webcam detection, and a REST API.

## 📌 Project Overview

This project builds and compares three deep learning models (Custom CNN, MobileNetV2, ResNet50) for binary face mask classification, then deploys the best model via a real-time OpenCV application and a FastAPI REST service, with an optional Streamlit frontend.

## 📂 Repository Structure

```
face-mask-detection/
│
├── dataset/                # Train/Validation/Test images (not committed - see Setup)
├── notebooks/               # Jupyter notebooks (EDA, experiments)
├── src/
│   ├── download_dataset.py  # Downloads dataset from Kaggle
│   ├── data_preprocessing.py# Image resizing, normalization, augmentation
│   ├── eda.py                # Exploratory Data Analysis
│   ├── models.py             # Custom CNN, MobileNetV2, ResNet50 architectures
│   ├── train.py               # Trains all 3 models
│   ├── evaluate.py            # Model evaluation & comparison report
│   └── realtime_detection.py  # Webcam-based real-time detection
├── api/
│   └── app.py                 # FastAPI REST API
├── frontend/
│   └── streamlit_app.py       # Optional Streamlit web UI
├── models/                    # Saved trained models (.h5)
├── results/                    # Plots, metrics, comparison reports
├── screenshots/
├── requirements.txt
└── README.md
```

## 📊 Dataset

**Face Mask ~12K Images Dataset** (Kaggle)
🔗 https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset

- ~12,000 images across 2 classes: `WithMask`, `WithoutMask`
- Pre-split into Train / Validation / Test folders

### Download the dataset

**Option A - via script (recommended):**
```bash
pip install kagglehub
python src/download_dataset.py
```

**Option B - manual:**
1. Download from the Kaggle link above (needs a free Kaggle account).
2. Extract and place folders so structure looks like:
```
dataset/
├── train/
│   ├── WithMask/
│   └── WithoutMask/
├── validation/
│   ├── WithMask/
│   └── WithoutMask/
└── test/
    ├── WithMask/
    └── WithoutMask/
```

## ⚙️ Installation

```bash
git clone https://github.com/hamzamunirml/face-mask-detection.git
cd face-mask-detection
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Download & organize dataset
```bash
python src/download_dataset.py
```

### 2. Exploratory Data Analysis
```bash
cd src
python eda.py
```

### 3. Train models
```bash
cd src
python train.py --model all           # trains all 3 models
python train.py --model mobilenetv2   # or train a single model
```

### 4. Evaluate & compare models
```bash
python evaluate.py
```

### 5. Real-time webcam detection
```bash
python realtime_detection.py --model ../models/mobilenetv2_best.h5
```
Press `q` to quit the webcam window.

### 6. Run the REST API
```bash
cd api
uvicorn app:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for interactive Swagger API docs.

**Example request:**
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@sample.jpg"
```

### 7. Run the Streamlit frontend (optional)
```bash
cd frontend
streamlit run streamlit_app.py
```

## 🧠 Model Architecture

| Model | Type | Params (approx) | Notes |
|---|---|---|---|
| Custom CNN | Built from scratch | ~2-4M | 4 Conv blocks + BatchNorm + Dropout |
| MobileNetV2 | Transfer Learning | ~2.3M | Lightweight, ideal for real-time/edge |
| ResNet50 | Transfer Learning | ~23.5M | Deeper, higher capacity |

All models use a sigmoid output for binary classification (Mask / No Mask).

## 📈 Results

Run `src/evaluate.py` after training to generate:
- Confusion matrices (`results/*_confusion_matrix.png`)
- ROC curves (`results/*_roc_curve.png`)
- Training history plots (`results/*_training_history.png`)
- `results/model_comparison_report.json` with Accuracy, Precision, Recall, F1-Score, AUC for each model

## 🔌 API Documentation

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/model-info` | Loaded model metadata |
| POST | `/predict` | Upload an image (`multipart/form-data`, field name `file`) → returns prediction + confidence |

**Response example:**
```json
{
  "prediction": "Mask",
  "confidence": 97.42,
  "raw_score": 0.0258
}
```

## 🌱 Future Improvements

- Add multi-face detection with individual bounding boxes and labels (already supported in `realtime_detection.py` via Haar Cascade loop)
- Export prediction logs to CSV
- Convert best model to TensorFlow Lite / ONNX for edge deployment
- Add Docker support and CI/CD via GitHub Actions
- Deploy live demo on Hugging Face Spaces / Render

## 🛠️ Tech Stack

Python · TensorFlow/Keras · OpenCV · FastAPI · Streamlit · scikit-learn · Pandas · NumPy · Matplotlib


