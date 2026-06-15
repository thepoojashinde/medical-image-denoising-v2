import os
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import scipy.io as sio
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from unet_model import unet_model


# ─────────────────────────────────────────
# FSIM
# ─────────────────────────────────────────

def fsim(img1, img2):
    def gradient_magnitude(img):
        img_uint8 = (img * 255).astype(np.uint8)
        gx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(gx**2 + gy**2)

    T1, T2 = 0.85, 160.0
    PC1 = gradient_magnitude(img1)
    PC2 = gradient_magnitude(img2)
    PCm = np.maximum(PC1, PC2)
    S_PC = (2 * PC1 * PC2 + T1) / (PC1**2 + PC2**2 + T1)
    S_G  = (2 * PC1 * PC2 + T2) / (PC1**2 + PC2**2 + T2)
    return np.sum(S_PC * S_G * PCm) / (np.sum(PCm) + 1e-8)


# ─────────────────────────────────────────
# DWT helpers
# ─────────────────────────────────────────

def apply_dwt(img_2d, wavelet='haar'):
    """Returns LL and (LH, HL, HH) subbands."""
    LL, (LH, HL, HH) = pywt.dwt2(img_2d, wavelet)
    return LL, (LH, HL, HH)


def apply_idwt(LL, high_freq_bands, wavelet='haar'):
    """Reconstructs full image from LL + high-freq bands via IDWT."""
    return pywt.idwt2((LL, high_freq_bands), wavelet)


def normalize_subband(band):
    """Normalize a subband to [0,1]. Returns normalized, min, max."""
    mn, mx = band.min(), band.max()
    return (band - mn) / (mx - mn + 1e-8), mn, mx


def denormalize_subband(band_norm, mn, mx):
    """Reverse the normalization."""
    return band_norm * (mx - mn + 1e-8) + mn


# ─────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────

def load_dicom(path, size=(128, 128)):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return cv2.resize(img, size)   # (128,128)


def add_gaussian_noise(img, mean=0, std=0.05):
    return np.clip(img + np.random.normal(mean, std, img.shape), 0, 1)


def add_poisson_noise(img):
    return np.clip(np.random.poisson(img * 255.0) / 255.0, 0, 1)


# ─────────────────────────────────────────
# Dataset builder
#
# INPUT:  Noisy LL subband  (64x64)   ← what U-Net receives
# OUTPUT: Clean LL subband  (64x64)   ← what U-Net must learn
#
# We also store noisy high-freq bands so IDWT can reconstruct later,
# but U-Net never sees or predicts them.
# ─────────────────────────────────────────

def build_dwt_dataset(dicom_paths, wavelet='haar'):
    X, Y = [], []

    for path in dicom_paths:
        try:
            clean = load_dicom(path)                        # (128,128)
            LL_clean, _ = apply_dwt(clean, wavelet)        # (64,64) — target

            for noise_fn in [add_gaussian_noise, add_poisson_noise]:
                noisy = noise_fn(clean)
                LL_noisy, _ = apply_dwt(noisy, wavelet)    # (64,64) — input

                # Normalize both subbands consistently
                LL_noisy_norm, _, _ = normalize_subband(LL_noisy)
                LL_clean_norm, _, _ = normalize_subband(LL_clean)

                X.append(LL_noisy_norm[..., np.newaxis])   # (64,64,1)
                Y.append(LL_clean_norm[..., np.newaxis])   # (64,64,1)

        except Exception as e:
            print(f"Skipping {path}: {e}")

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return shuffle(X, Y, random_state=42)


# ─────────────────────────────────────────
# Full denoising pipeline
#
# noisy (128x128)
#   → DWT → Noisy LL + Noisy {LH, HL, HH}
#   → U-Net on Noisy LL → Clean LL
#   → IDWT(Clean LL, Noisy high-freq) → Final image (128x128)
# ─────────────────────────────────────────

def denoise_image(model, noisy_2d, wavelet='haar'):
    # Step 1: Decompose noisy image
    LL_noisy, (LH, HL, HH) = apply_dwt(noisy_2d, wavelet)

    # Step 2: Normalize LL for model input, save stats to denormalize later
    LL_norm, ll_min, ll_max = normalize_subband(LL_noisy)

    # Step 3: U-Net predicts clean LL (still in [0,1] normalized space)
    inp = LL_norm[np.newaxis, ..., np.newaxis]             # (1,64,64,1)
    LL_clean_norm = model.predict(inp, verbose=0)[0, :, :, 0]  # (64,64)

    # Step 4: Denormalize back to original LL scale
    LL_clean = denormalize_subband(LL_clean_norm, ll_min, ll_max)

    # Step 5: IDWT — clean LL + original noisy high-freq bands → (128,128)
    # Note: using noisy high-freq bands is standard; they're less affected
    # by Gaussian noise than LL. You can also threshold them here later.
    reconstructed = apply_idwt(LL_clean, (LH, HL, HH), wavelet)

    return np.clip(reconstructed, 0, 1)


# ─────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────

def compute_metrics(clean, denoised):
    mse_val  = mean_squared_error(clean.flatten(), denoised.flatten())
    psnr_val = psnr(clean, denoised, data_range=1.0)
    ssim_val = ssim(clean, denoised, data_range=1.0)
    fsim_val = fsim(clean, denoised)
    return mse_val, psnr_val, ssim_val, fsim_val


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Step 1: Collect DICOM paths
    dicom_paths = []
    for root, _, files in os.walk('dataset/manifest-1600709154662/LIDC-IDRI'):
        for f in files:
            if f.endswith('.dcm'):
                dicom_paths.append(os.path.join(root, f))
    print(f"Found {len(dicom_paths)} DICOM files.")

    # Step 2: Build dataset (LL_noisy → LL_clean, both 64x64)
    print("Building DWT dataset...")
    X, Y = build_dwt_dataset(dicom_paths)
    print(f"X (noisy LL): {X.shape}, Y (clean LL): {Y.shape}")

    # Step 3: Train or load
    MODEL_PATH = "unet_dwt_v3_model.keras"

    if os.path.exists(MODEL_PATH):
        print("Loading pre-trained model...")
        model = load_model(MODEL_PATH)
    else:
        print("Training from scratch...")
        model = unet_model(input_size=(64, 64, 1))
        model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
        model.summary()
        model.fit(X, Y, epochs=50, batch_size=16, validation_split=0.1)
        model.save(MODEL_PATH)
        print(f"Model saved as '{MODEL_PATH}'")

    # Step 4: Evaluate on external images
    external_paths = [
        "lidc_subject_0001_slice_40.dcm",
        "lidc_subject_0023_slice_26.dcm",
        "1-4.dcm"
    ]

    output_dir = "result_dwt_v3"
    os.makedirs(output_dir, exist_ok=True)

    std_values = np.linspace(0.005, 0.05, 10)
    all_dfs = []

    for idx, path in enumerate(external_paths):
        image_id = f"img{idx+1}"
        try:
            clean = load_dicom(path)
            print(f"[✓] Loaded {image_id}")
        except Exception as e:
            print(f"[!] Skipping {path}: {e}")
            continue

        rows = []

        for std in std_values:
            noisy    = add_gaussian_noise(clean, std=std)
            denoised = denoise_image(model, noisy)

            mse_val, psnr_val, ssim_val, fsim_val = compute_metrics(clean, denoised)
            rows.append({
                "STD":       round(std, 3),
                "MSE":       round(mse_val, 6),
                "PSNR (dB)": round(psnr_val, 2),
                "SSIM":      round(ssim_val, 6),
                "FSIM":      round(fsim_val, 6)
            })

            np.save(os.path.join(output_dir, f"{image_id}_noisy_std_{std:.3f}.npy"), noisy)
            sio.savemat(os.path.join(output_dir, f"{image_id}_noisy_std_{std:.3f}.mat"),
                        {f"noisy_std_{std:.3f}": noisy})

            if np.isclose(std, 0.05):
                np.save(os.path.join(output_dir, f"{image_id}_clean.npy"), clean)
                np.save(os.path.join(output_dir, f"{image_id}_denoised_std_0.05.npy"), denoised)
                sio.savemat(os.path.join(output_dir, f"{image_id}_std_0.05_comparison.mat"),
                            {"clean": clean, "noisy": noisy, "denoised": denoised})

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                for ax, img, title in zip(axes,
                                          [clean, noisy, denoised],
                                          ["Clean", "Noisy (std=0.05)", "Denoised (DWT+UNet+IDWT)"]):
                    ax.imshow(img, cmap='gray')
                    ax.set_title(title)
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{image_id}_comparison.png"), dpi=300)
                plt.close()

        df = pd.DataFrame(rows)
        df.insert(0, "Image", image_id)
        df.to_csv(os.path.join(output_dir, f"{image_id}_robustness_table.csv"), index=False)
        all_dfs.append(df)
        print(f"\nRobustness table for {image_id}:")
        print(df.to_string(index=False))

    if all_dfs:
        combined = pd.concat(all_dfs)
        avg = combined.groupby("STD")[["MSE","PSNR (dB)","SSIM","FSIM"]].mean().round(6)
        avg.to_csv(os.path.join(output_dir, "average_robustness_table.csv"))
        print("\nAverage across all images:")
        print(avg.to_string())

    print(f"\n[✓] All results saved to '{output_dir}/'")