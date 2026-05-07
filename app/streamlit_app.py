from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_classification.inference import predict_image

st.set_page_config(page_title="TrashNet Classifier", layout="centered")
st.title("TrashNet Image Classification Demo")
st.write("Upload an image to predict the waste category with the trained model.")

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
