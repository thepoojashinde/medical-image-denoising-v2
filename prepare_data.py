import os
import numpy as np
from tqdm import tqdm
from dicombasics import dicom_processor  # or from fndicom if your class is there

# Paths
base_dir = "dataset"  # Folder where .dcm files are

X = []
Y = []

for file in tqdm(os.listdir(base_dir)):
    if file.endswith('.dcm'):
        path = os.path.join(base_dir, file)
        dcm = dicom_processor(path)

        clean = dcm.get_clean_image()
        noisy = dcm.get_noisy_image(noise_type='gaussian')  # or 'poisson'

        if clean is not None and noisy is not None:
            X.append(dcm.prepare_for_model(noisy))
            Y.append(dcm.prepare_for_model(clean))

# Convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# Save to .npy files
np.save(os.path.join(base_dir, 'X.npy'), X)
np.save(os.path.join(base_dir, 'Y.npy'), Y)

print("✅ Dataset processed and saved.")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")
