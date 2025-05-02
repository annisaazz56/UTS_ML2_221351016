import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

# --- LOAD MODEL DAN SCALER ---
try:
    base_path = os.path.dirname(os.path.abspath(__file__))
    model = tf.keras.models.load_model("drybean_model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    st.warning("Model/scaler/label_encoder gak ketemu. Prediksi random aja ya! 🤷‍♂️")
    st.error(f"Detail error: {e}")  # <<< tampilkan error asli di Streamlit
    model = None
    scaler = None
    label_encoder = None

# --- FITUR SESUAI DATASET DRY BEAN ---
feature_names = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
    "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
    "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2",
    "ShapeFactor3", "ShapeFactor4"
]

# --- HEADER ---
st.markdown(
    """
    <h1 style='text-align:center; color:#7F56D9; font-family:monospace;'>
        🫘 Kacang Checker 2K25 🚀
    </h1>
    <p style='text-align:center; color:#666; font-size:1.2rem;'>
        Isi ciri-ciri kacangmu, <b>klik prediksi</b>, langsung tau jenisnya!<br>
        <span style='font-size:1.5rem;'>#GenZStyle</span>
    </p>
    <hr>
    """, unsafe_allow_html=True
)

# --- INPUT FITUR DALAM 2 KOLOM ---
st.subheader("Masukkan ciri-ciri kacangmu di sini 👇")
col1, col2 = st.columns(2)
input_data = []
for i, feature in enumerate(feature_names):
    with (col1 if i % 2 == 0 else col2):
        value = st.number_input(f"{feature}", value=0.0, format="%.4f")
        input_data.append(value)

# --- TOMBOL PREDIKSI ---
if st.button("✨ Prediksi Sekarang!"):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    st.caption("Input kamu:")
    st.dataframe(input_df, hide_index=True, use_container_width=True)

    if scaler is not None:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df.values

    if model is not None:
        pred_proba = model.predict(input_scaled)
        pred_class_index = np.argmax(pred_proba, axis=1)
        if label_encoder is not None:
            pred_label = label_encoder.inverse_transform(pred_class_index)[0]
        else:
            pred_label = str(pred_class_index[0])
    else:
        possible_classes = ["SEKER", "DERMASON", "BARBUNYA", "BOMBAY", "CALI", "HOROZ", "SIRA"]
        pred_label = np.random.choice(possible_classes)

    emoji = "🫘"
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align:center; padding:2rem 0;'>
            <span style='font-size:2.5rem;'>{emoji}</span>
            <h2 style='color:#00C897; font-family:monospace;'>
                Jenis kacangmu: <span style='font-size:2.5rem'>{pred_label}</span>
            </h2>
            <span style='font-size:1.5rem;'>#StayNutty</span>
        </div>
        """, unsafe_allow_html=True
    )
    st.snow()
