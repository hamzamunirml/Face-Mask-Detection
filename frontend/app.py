"""
Phase 10 - Frontend Development
Upgraded Streamlit interface for Face Mask Detection.

Run:
    streamlit run streamlit_app.py
"""

import os
import time
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ----------------------------- CONFIG -----------------------------

IMG_SIZE = (128, 128)
LABELS = {0: "Mask", 1: "No Mask"}
EMOJIS = {0: "😷", 1: "⚠️"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_OPTIONS = {
    "MobileNetV2 (recommended)": os.path.join(BASE_DIR, "..", "models", "mobilenetv2_best.keras"),
    "Custom CNN": os.path.join(BASE_DIR, "..", "models", "custom_cnn_best.keras"),
    "ResNet50": os.path.join(BASE_DIR, "..", "models", "resnet50_best.keras"),
}

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------- STYLE -----------------------------

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem;}
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
    }
    .mask-card {background-color: #16342b; border: 1px solid #2ecc71;}
    .nomask-card {background-color: #3a1414; border: 1px solid #e74c3c;}
    .result-label {font-size: 1.8rem; font-weight: 700; margin-bottom: 0.2rem;}
    .result-conf {font-size: 1rem; opacity: 0.85;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- HELPERS -----------------------------

@st.cache_resource(show_spinner=False)
def get_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    return load_model(model_path)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image):
    processed = preprocess(image)
    pred_prob = float(model.predict(processed, verbose=0)[0][0])
    label_idx = int(pred_prob >= 0.5)
    confidence = pred_prob if label_idx == 1 else 1 - pred_prob
    return label_idx, confidence


def render_result(label_idx: int, confidence: float):
    css_class = "nomask-card" if label_idx == 1 else "mask-card"
    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="result-label">{EMOJIS[label_idx]} {LABELS[label_idx]}</div>
            <div class="result-conf">Confidence: {confidence*100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(confidence)


# ----------------------------- SIDEBAR -----------------------------

st.sidebar.title("⚙️ Settings")

model_choice = st.sidebar.selectbox("Select model", list(MODEL_OPTIONS.keys()))
model_path = MODEL_OPTIONS[model_choice]

threshold = st.sidebar.slider(
    "Decision threshold (No Mask if prob ≥ this)", 0.0, 1.0, 0.5, 0.01
)

input_mode = st.sidebar.radio("Input source", ["Upload image", "Use camera"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: Transfer Learning (MobileNetV2 / ResNet50) or Custom CNN\n\n"
    "Trained on Face Mask 12K Dataset"
)

if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------- MAIN -----------------------------

st.title("😷 Real-Time Face Mask Detection")
st.write("Upload an image or use your camera to check whether a person is wearing a mask.")

model = get_model(model_path)

if model is None:
    st.error(
        f"Model file not found at `{model_path}`. "
        "Please check the path in `MODEL_OPTIONS` or train/export the model first."
    )
    st.stop()

image = None
if input_mode == "Upload image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        image = Image.open(camera_file)

if image is not None:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)

    with col2:
        if st.button("🔍 Predict", use_container_width=True):
            with st.spinner("Analyzing..."):
                pred_prob = float(
                    model.predict(preprocess(image), verbose=0)[0][0]
                )
                label_idx = int(pred_prob >= threshold)
                confidence = pred_prob if label_idx == 1 else 1 - pred_prob
                time.sleep(0.2)  # smoother UX for the spinner

            render_result(label_idx, confidence)

            st.session_state.history.insert(
                0,
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "model": model_choice,
                    "result": LABELS[label_idx],
                    "confidence": f"{confidence*100:.2f}%",
                },
            )
        else:
            st.info("Click **Predict** to run detection on this image.")

# ----------------------------- HISTORY -----------------------------

if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Prediction History (this session)")
    st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)

    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

st.markdown("---")
st.caption("Face Mask Detection App | Built with Streamlit + TensorFlow/Keras")
