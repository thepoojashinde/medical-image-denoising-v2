# Medical Image Denoising using DWT and U-Net

## Overview

This project presents a deep learning approach for medical image denoising using Discrete Wavelet Transform (DWT) and a modified 4-channel U-Net architecture. The model is trained on CT scan images from the LIDC-IDRI dataset with synthetically added Gaussian noise.

## Features

- DWT-based preprocessing
- Custom 4-channel U-Net architecture
- Comparison with BM3D, DnCNN, and ResUNet
- Performance evaluation using:
  - PSNR
  - SSIM
  - FSIM
  - MSE

## Dataset

- LIDC-IDRI CT Scan Dataset

## Technologies

- Python
- TensorFlow / Keras
- OpenCV
- PyWavelets
- pydicom
- NumPy
- Matplotlib

## Repository Structure

- Medical_Image_Denoising_DWT_UNet.ipynb — Complete training pipeline
- train_dwt_unet.py — Training script
- unet_model_4ch.py — Proposed model architecture
- requirements.txt — Required dependencies

## Future Work

- Deploy the model using Streamlit for real-time CT image denoising.
