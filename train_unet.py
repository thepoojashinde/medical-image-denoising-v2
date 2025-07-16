import os
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from unet_model import unet_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import scipy.io as sio  # For saving .mat files

# -------------------------
# Utility Functions
# -------------------------

def load_dicom(path, size=(128, 128)):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # normalize
    img = cv2.resize(img, size)
    return img[..., np.newaxis]


def add_gaussian_noise(img, mean=0, std=0.05):
    noise = np.random.normal(mean, std, img.shape)
    noisy = img + noise
    noisy = np.clip(noisy, 0, 1)
    return noisy


def add_poisson_noise(image):
    noisy = np.random.poisson(image * 255.0) / 255.0
    noisy = np.clip(noisy, 0, 1)
    return noisy


# -------------------------
# Step 1: Load DICOM Images
# -------------------------

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

        X.append(noisy_gaussian)
        Y.append(clean)

        X.append(noisy_poisson)
        Y.append(clean)
    except Exception as e:
        print(f"Error processing {path}: {e}")

X = np.array(X)
Y = np.array(Y)
X, Y = shuffle(X, Y, random_state=42)

print("X shape:", X.shape)
print("Y shape:", Y.shape)


# -------------------------
# Step 2: Train or Load Model
# -------------------------

if os.path.exists("unet_denoising_model.keras"):
    print("Loading pre-trained model...")
    model = load_model("unet_denoising_model.keras")
else:
    print("Training model from scratch...")
    model = unet_model()
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    model.fit(X, Y, epochs=50, batch_size=8)
    model.save("unet_denoising_model.keras")
    print("Model saved as 'unet_denoising_model.keras'")


# -------------------------
# Step 3: Visualize One Example
# -------------------------

i = 0
pred = model.predict(np.expand_dims(X[i], axis=0))[0]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Noisy")
plt.imshow(X[i][:, :, 0], cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Denoised")
plt.imshow(pred[:, :, 0], cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Clean")
plt.imshow(Y[i][:, :, 0], cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


# -------------------------
# Step 4: Evaluate Model
# -------------------------

def evaluate_model(model, X, Y):
    ssim_list = []
    mse_list = []
    psnr_list = []

    for i in range(len(X)):
        noisy_img = X[i]
        clean_img = Y[i]
        denoised_img = model.predict(np.expand_dims(noisy_img, axis=0), verbose=0)[0]

        noisy_img = noisy_img[:, :, 0]
        clean_img = clean_img[:, :, 0]
        denoised_img = denoised_img[:, :, 0]

        mse_val = mean_squared_error(clean_img, denoised_img)
        ssim_val = ssim(clean_img, denoised_img, data_range=1.0)
        psnr_val = psnr(clean_img, denoised_img, data_range=1.0)

        mse_list.append(mse_val)
        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)

    print("\nAverage SSIM:", round(np.mean(ssim_list),6))
    print("Average MSE:", round(np.mean(mse_list),2))
    print("Average PSNR:", round(np.mean(psnr_list),6))


try:
    evaluate_model(model, X, Y)
except Exception as e:
    print("Error in evaluation:", e)


# -------------------------
# Step 5: Predict on 3 External Images and Save
# -------------------------

external_paths = [
    "lidc_subject_0001_slice_40.dcm",
    "lidc_subject_0023_slice_26.dcm",
    "1-4.dcm"
]

output_dir = "result"
os.makedirs(output_dir, exist_ok=True)

clean_list = []
noisy_list_005 = []
denoised_list_005 = []

std_list = np.linspace(0.005, 0.05, 10)

for idx, path in enumerate(external_paths):
    try:
        clean_img = load_dicom(path)
        print(f"[✓] Loaded image {idx+1}")

        for std_val in std_list:
            noisy_img = add_gaussian_noise(clean_img, std=std_val)

            # Save noisy as .npy
            np.save(os.path.join(output_dir, f"img{idx+1}_noisy_std_{std_val:.3f}.npy"), noisy_img)

            # Save noisy as .mat
            sio.savemat(os.path.join(output_dir, f"img{idx+1}_noisy_std_{std_val:.3f}.mat"), {
                f"noisy_std_{std_val:.3f}": noisy_img[:, :, 0]
            })

            # For std = 0.05 → also denoise
            if np.isclose(std_val, 0.05):
                denoised_img = model.predict(np.expand_dims(noisy_img, axis=0), verbose=0)[0]

                # Save clean + denoised
                np.save(os.path.join(output_dir, f"img{idx+1}_clean.npy"), clean_img)
                np.save(os.path.join(output_dir, f"img{idx+1}_denoised_std_0.05.npy"), denoised_img)

                sio.savemat(os.path.join(output_dir, f"img{idx+1}_std_0.05_comparison.mat"), {
                    "clean": clean_img[:, :, 0],
                    "noisy": noisy_img[:, :, 0],
                    "denoised": denoised_img[:, :, 0]
                })

                # Append for visual table
                clean_list.append(clean_img)
                noisy_list_005.append(noisy_img)
                denoised_list_005.append(denoised_img)

    except Exception as e:
        print(f"[!] Skipping {path} due to error: {e}")
        continue


# -------------------------
# Step 6: Noise Robustness Evaluation Table (10 stds)
# -------------------------

print("\nStep 6: Evaluating model for different Gaussian noise std values...")

std_values = np.linspace(0.005, 0.05, 10).round(5).tolist()
test_paths = dicom_paths[-10:]

results = []

for std in std_values:
    mse_list, ssim_list, psnr_list = [], [], []

    for path in test_paths:
        try:
            clean = load_dicom(path)
            noisy = add_gaussian_noise(clean, std=std)
            denoised = model.predict(np.expand_dims(noisy, axis=0), verbose=0)[0]

            c = clean[:, :, 0]
            d = denoised[:, :, 0]

            mse_list.append(mean_squared_error(c, d))
            ssim_list.append(ssim(c, d, data_range=1.0))
            psnr_list.append(psnr(c, d, data_range=1.0))

        except Exception as e:
            print(f"Error processing {path}: {e}")

    results.append({
        "Noise Std": std,
        "MSE": round(np.mean(mse_list), 6),
        "PSNR (dB)": round(np.mean(psnr_list), 2),
        "SSIM": round(np.mean(ssim_list), 6)
    })

df = pd.DataFrame(results)
print("\n Noise Robustness Evaluation Table:\n")
print(df.to_string(index=False))


def plot_final_comparison(clean_list, noisy_list, denoised_list, save_path):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    titles = ["Clean", "Noisy (std=0.05)", "Denoised"]
    for i in range(3):
        images = [clean_list[i], noisy_list[i], denoised_list[i]]
        for j in range(3):
            axes[i][j].imshow(images[j][:,:,0], cmap="gray")
            axes[i][j].axis("off")
            if i == 0:
                axes[i][j].set_title(titles[j])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Final comparison table saved to {save_path}")


if len(clean_list) == 3 and len(noisy_list_005) == 3 and len(denoised_list_005) == 3:
    plot_final_comparison(
        clean_list,
        noisy_list_005,
        denoised_list_005,
        os.path.join(output_dir, "comparison_std_0.05.png")
    )
    print("[✓] Visual comparison table saved.")
else:
    print(f"[!] Skipping plot: Not enough images. Got {len(clean_list)}/3")


# -------------------------
# Step 7: Evaluation Tables for img1, img2, img3 at 10 std values
# -------------------------

def evaluate_image_at_different_std(image, std_values, image_id):
    results = []

    for std in std_values:
        try:
            noisy = add_gaussian_noise(image, std=std)
            denoised = model.predict(np.expand_dims(noisy, axis=0), verbose=0)[0]

            clean = image[:, :, 0]
            den = denoised[:, :, 0]

            mse_val = mean_squared_error(clean, den)
            ssim_val = ssim(clean, den, data_range=1.0)
            psnr_val = psnr(clean, den, data_range=1.0)

            results.append({
                "Image": image_id,
                "Noise Std": round(std, 5),
                "MSE": round(mse_val, 6),
                "PSNR (dB)": round(psnr_val, 2),
                "SSIM": round(ssim_val, 6)
            })

        except Exception as e:
            print(f"[!] Error processing {image_id} with std={std}: {e}")

    return pd.DataFrame(results)

# Gaussian noise std values from 0.005 to 0.050
std_values = np.linspace(0.005, 0.05, 10)

# Load the external images
try:
    img1_clean = load_dicom(external_paths[0])
    img2_clean = load_dicom(external_paths[1])
    img3_clean = load_dicom(external_paths[2])
except Exception as e:
    print("[!] Error loading external images:", e)
    raise

# Evaluate each image
df_img1 = evaluate_image_at_different_std(img1_clean, std_values, "img1")
df_img2 = evaluate_image_at_different_std(img2_clean, std_values, "img2")
df_img3 = evaluate_image_at_different_std(img3_clean, std_values, "img3")

# Print all three tables
print("\n📊 Noise Robustness Table for img1:\n")
print(df_img1.to_string(index=False))

print("\n📊 Noise Robustness Table for img2:\n")
print(df_img2.to_string(index=False))

print("\n📊 Noise Robustness Table for img3:\n")
print(df_img3.to_string(index=False))

# Save to CSV
df_img1.to_csv(os.path.join(output_dir, "img1_robustness_table.csv"), index=False)
df_img2.to_csv(os.path.join(output_dir, "img2_robustness_table.csv"), index=False)
df_img3.to_csv(os.path.join(output_dir, "img3_robustness_table.csv"), index=False)

print("\n[✓] All robustness tables saved to 'result/' folder.")

