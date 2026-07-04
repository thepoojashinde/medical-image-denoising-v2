import cv2
import numpy as np
import pydicom
import pywt

def load_dicom(path, size=(128, 128)):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return cv2.resize(img, size)

def add_gaussian_noise(img, mean=0, std=0.05):
    return np.clip(img + np.random.normal(mean, std, img.shape), 0, 1)

def apply_dwt(img_2d, wavelet='haar'):
    LL, (LH, HL, HH) = pywt.dwt2(img_2d, wavelet)
    return LL, (LH, HL, HH)

def apply_idwt(LL, high_freq_bands, wavelet='haar'):
    return pywt.idwt2((LL, high_freq_bands), wavelet)
