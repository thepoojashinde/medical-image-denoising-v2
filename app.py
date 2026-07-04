import streamlit as st
import numpy as np
import io
from PIL import Image

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

    # Load image
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "dcm":
        original = load_dicom(uploaded_file)
    else:
        original = load_image(uploaded_file)

    # Add optional Gaussian noise
    noisy = original.copy()

    if demo_mode:
        noisy = add_gaussian_noise(original, std=noise_std)

    # Create layout
    col1, col2, col3 = st.columns(3)

    # Show uploaded image
    with col1:
        st.subheader("Uploaded Image")
        st.image(
            original,
            clamp=True,
            use_container_width=True
        )

    # Show model input
    with col2:
        if demo_mode:
            st.subheader("Noisy Input")
        else:
            st.subheader("Input to Model")

        st.image(
            noisy,
            clamp=True,
            use_container_width=True
        )

    # Button
    if st.button("🧠 Denoise Image"):

        with st.spinner("Running U-Net model..."):
            denoised = denoise_image(model, noisy)

        with col3:
            st.subheader("Denoised Image")

            st.image(
                denoised,
                clamp=True,
                use_container_width=True
            )

            # Download button
            denoised_uint8 = (denoised * 255).astype(np.uint8)
            pil_image = Image.fromarray(denoised_uint8)

            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")

            st.download_button(
                label="📥 Download Denoised Image",
                data=buffer.getvalue(),
                file_name="denoised_ct_scan.png",
                mime="image/png"
            )

    
