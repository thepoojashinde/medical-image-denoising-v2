import cv2
import numpy as np
import pydicom
import pywt
from PIL import Image
from tensorflow.keras.models import load_model

def load_dicom(path, size=(128, 128)):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return cv2.resize(img, size)

def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((128, 128))
    img = np.array(img, dtype=np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def add_gaussian_noise(img, mean=0, std=0.05):
    return np.clip(img + np.random.normal(mean, std, img.shape), 0, 1)

def apply_dwt(img_2d, wavelet='haar'):
    LL, (LH, HL, HH) = pywt.dwt2(img_2d, wavelet)
    return LL, (LH, HL, HH)

def apply_idwt(LL, high_freq_bands, wavelet='haar'):
    return pywt.idwt2((LL, high_freq_bands), wavelet)

def preprocess_image(img):
    """
    Converts a normalized 128x128 image into a
    64x64x4 DWT tensor ready for the model.
    """

    LL, (LH, HL, HH) = apply_dwt(img)

    x = np.stack([LL, LH, HL, HH], axis=-1)
    x = np.expand_dims(x, axis=0)

    return x

def denoise_image(model, img):
    """
    Performs denoising using the trained U-Net model.
    Returns the reconstructed denoised image.
    """

    # Preprocess
    x = preprocess_image(img)

    # Predict
    pred = model.predict(x, verbose=0)[0]

    # Separate channels
    LL = pred[:, :, 0]
    LH = pred[:, :, 1]
    HL = pred[:, :, 2]
    HH = pred[:, :, 3]

    # Reconstruct image
    denoised = apply_idwt(LL, (LH, HL, HH))

    # Keep values between 0 and 1
    denoised = np.clip(denoised, 0, 1)

    return denoised

def load_trained_model():
    model = load_model("unet_dwt_v4_model.keras")
    return model


