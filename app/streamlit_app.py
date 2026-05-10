from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"
SRC_DIR = ROOT / "src"
METRICS_PATH = ROOT / "artifacts" / "reports" / "metrics.json"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_classification.inference import predict_image


def load_metrics(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))

st.set_page_config(page_title="TrashNet Classifier", layout="centered")
st.title("TrashNet Image Classification Demo")
st.write("Upload an image to predict the waste category with the trained model.")

metrics = load_metrics(METRICS_PATH)
if metrics is not None:
    st.subheader("Model Evaluation Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Top-1 Accuracy", f"{metrics['top1_accuracy']:.2f}%")
    col2.metric("Top-5 Accuracy", f"{metrics['top5_accuracy']:.2f}%")
    col3.metric("Overall Accuracy", f"{metrics['overall_accuracy']:.2f}%")
else:
    st.info("Evaluation metrics are not available yet. Run the evaluation stage to generate them.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input image", use_container_width=True)

    predictions = predict_image(image=image, params_path=PARAMS_PATH)
    top_label, top_score = predictions[0]

    st.subheader("Prediction")
    st.write(f"Top class: **{top_label}**")
    st.write(f"Model score: **{top_score:.2%}**")

    st.subheader("Top scores")
    st.table(
        [{"class": label, "score": f"{score:.2%}"} for label, score in predictions[:5]]
    )
