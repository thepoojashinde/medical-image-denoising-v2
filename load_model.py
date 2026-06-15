import os
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error

# ---------------------------
# Load and preprocess DICOM image
# ---------------------------
def load_dicom(path, size=(128, 128)):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # normalize safely
    img = cv2.resize(img, size)
    return img[..., np.newaxis]  # add channel dimension

def add_gaussian_noise(img, mean=0, std=0.05):
    noise = np.random.normal(mean, std, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 1)

def add_poisson_noise(image):
    noisy = np.random.poisson(image * 255.0) / 255.0
    return np.clip(noisy, 0, 1)

# ---------------------------
# Load dataset
# ---------------------------
dicom_paths = []
for root, _, files in os.walk('dataset/manifest-1600709154662/LIDC-IDRI'):
    for f in files:
        if f.endswith('.dcm'):
            dicom_paths.append(os.path.join(root, f))

print(f"Found {len(dicom_paths)} DICOM files.")

X, Y = [], []

for path in dicom_paths:
    try:
        clean = load_dicom(path)
        noisy_gaussian = add_gaussian_noise(clean)
        noisy_poisson = add_poisson_noise(clean)

        X.extend([noisy_gaussian, noisy_poisson])
        Y.extend([clean, clean])
    except Exception as e:
        print(f"Error processing {path}: {e}")

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# ---------------------------
# Load the pre-trained U-Net model (safe load)
# ---------------------------
print("Loading model...")

try:
    # First try normal load
    model = load_model("unet_denoising_model.h5")
except TypeError:
    # Handle Keras 3.x 'mse' issue
    model = load_model("unet_denoising_model.h5", custom_objects={'mse': MeanSquaredError()})
except ValueError:
    # If still fails, skip compilation
    model = load_model("unet_denoising_model.h5", compile=False)

print("Model loaded successfully!")

# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate_model(model, X, Y):
    ssim_list, mse_list, psnr_list = [], [], []

    for i in range(len(X)):
        noisy_img = X[i]
        clean_img = Y[i]
        denoised_img = model.predict(np.expand_dims(noisy_img, axis=0), verbose=0)[0]

        noisy_img = noisy_img[:, :, 0]
        clean_img = clean_img[:, :, 0]
        denoised_img = np.clip(denoised_img[:, :, 0], 0, 1)

        mse_val = mean_squared_error(clean_img, denoised_img)
        ssim_val = ssim(clean_img, denoised_img, data_range=1.0)
        psnr_val = psnr(clean_img, denoised_img, data_range=1.0)

        mse_list.append(mse_val)
        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)

        if i < 3:  # show a few
            print(f"[{i}] MSE: {mse_val:.4f}, SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB")

    print("\nAverage SSIM:", np.mean(ssim_list))
    print("Average MSE:", np.mean(mse_list))
    print("Average PSNR:", np.mean(psnr_list))

# ---------------------------
# Prediction on New DICOM Images
# ---------------------------
new_image_paths = [
    "lidc_subject_0001_slice_40.dcm",
    "lidc_subject_0023_slice_26.dcm"
]

def predict_on_new_images(model, paths):
    for idx, path in enumerate(paths):
        try:
            clean_img = load_dicom(path)
            noisy_img = add_gaussian_noise(clean_img)
            denoised_img = model.predict(np.expand_dims(noisy_img, axis=0), verbose=0)[0]

            clean_2d = clean_img[:, :, 0]
            denoised_2d = np.clip(denoised_img[:, :, 0], 0, 1)

            mse_val = mean_squared_error(clean_2d, denoised_2d)
            ssim_val = ssim(clean_2d, denoised_2d, data_range=1.0)
            psnr_val = psnr(clean_2d, denoised_2d, data_range=1.0)

            print(f"\nImage {idx+1} — MSE: {mse_val:.4f}, SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB")

            # Visualization
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1); plt.title("Noisy"); plt.imshow(noisy_img[:, :, 0], cmap='gray'); plt.axis('off')
            plt.subplot(1, 3, 2); plt.title("Denoised"); plt.imshow(denoised_2d, cmap='gray'); plt.axis('off')
            plt.subplot(1, 3, 3); plt.title("Clean"); plt.imshow(clean_2d, cmap='gray'); plt.axis('off')
            plt.suptitle(f"Prediction on Image {idx + 1}")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error processing {path}: {e}")

# ---------------------------
# Run evaluation and prediction
# ---------------------------
print("\nEvaluating model on training data...")
evaluate_model(model, X, Y)

print("\nRunning prediction on new DICOM images...")
predict_on_new_images(model, new_image_paths)
