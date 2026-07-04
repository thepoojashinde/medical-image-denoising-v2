import streamlit as st
import numpy as np

from utils import (
    load_dicom,
    load_image,
    add_gaussian_noise,
    load_trained_model,
    denoise_image,
)

st.set_page_config(
    page_title="Medical Image Denoising",
    layout="wide"
)

st.title("🩺 Medical Image Denoising using DWT + U-Net")
st.write("Upload a CT scan and view the denoised result.")

@st.cache_resource
def get_model():
    return load_trained_model()

model = get_model()

uploaded_file = st.file_uploader(
    "Upload a CT Scan",
    type=["dcm", "png", "jpg", "jpeg"]
)

demo_mode = st.checkbox(
    "Enable Demo Mode (Add Gaussian Noise)",
    value=False
)

noise_std = 0.05

if demo_mode:
    noise_std = st.slider(
        "Noise Standard Deviation",
        min_value=0.01,
        max_value=0.09,
        value=0.05,
        step=0.01
    )

if uploaded_file is not None:

    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "dcm":
        original = load_dicom(uploaded_file)
    else:
        original = load_image(uploaded_file)

    noisy = original.copy()

    if demo_mode:
        noisy = add_gaussian_noise(original, std=noise_std)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(original, clamp=True)

    with col2:
        st.subheader("Input to Model")
        st.image(noisy, clamp=True)

    st.markdown("---")

    if st.button("🧠 Denoise Image"):

        with st.spinner("Running U-Net model..."):

            denoised = denoise_image(model, noisy)

        st.subheader("Denoised Image")

        st.image(
            denoised,
            clamp=True,
            use_container_width=True
        )
