import matplotlib.pyplot as plt
import pydicom as dicom
import numpy as np
import cv2
import math
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from unet_model import unet_model



#ds holds the entire file (image + metadata)
#metadata - xray info
ds = dicom.dcmread('lidc_subject_0001_slice_40.dcm')
ds2 = dicom.dcmread('lidc_subject_0023_slice_26.dcm')

#printing few items from metadata
print(ds.PatientName)
print(ds.Modality)
print(ds.StudyDate)
print(ds.Rows,ds.Columns)

print(ds2.PatientName)
print(ds2.Modality)
print(ds2.StudyDate)
print(ds2.Rows,ds2.Columns)


#list of metadata
#print(dir(ds))

#access image(pixel data)
pixel_array = ds.pixel_array
print(pixel_array.shape)

pixel_array2 = ds2.pixel_array
print(pixel_array2.shape)

#convert to uint8 for openCV

#we convert our image to float32 because we need normalise aage jake so float is better
img = ds.pixel_array.astype(np.float32)

#here, we are normalizing to get in range [0,255], cv2.normalize does this on its own, cv2.norm_minmax uses minmax to normalize
#parameter - src, dst, min, max, type of normalisation
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

#we convert float to uint8 to use other lib - now we have our normal grayscale image
img_uint8 = img.astype(np.uint8)

img2 = ds2.pixel_array.astype(np.float32)
img2 = cv2.normalize(img2,None,0,255,cv2.NORM_MINMAX)
img2_uint8 = img2.astype(np.uint8)


#adding gaussian noise
#mean - brightness, std_Dev - intensity of noise
mean = 0 #constant brightness
std_dev = 20 #quite noisy

#generating a random noise array of same shape as img_uint8
gaussian_noise = np.random.normal(mean,std_dev,img_uint8.shape)

# added noise with image converted to float32 to wrap all the values perfectly
gnoisy1 = img_uint8.astype(np.float32)  + gaussian_noise

#clipping all the values btw 0 to 255 and converting to uint8 so that we can be displayed properly
gnoisy1 = np.clip(gnoisy1, 0, 255).astype(np.uint8)

gaussian_noise2 = np.random.normal(mean,std_dev,img2_uint8.shape)
gnoisy2 = img2_uint8.astype(np.float32) + gaussian_noise2
gnoisy2 = np.clip(gnoisy2,0,255).astype(np.uint8)

#adding poisson noise

#counting all the unique pixel intensities int the img_uint8
vals1 = len(np.unique(img_uint8))

#converting it to nearest 2's power as it is easier to process in np.random.poisson()
vals1 = 2**np.ceil(np.log2(vals1))

#image ko scale up kiya pehle noise add kiya and then scale down kiya
pnoisy1 = np.random.poisson(img_uint8*vals1)/float(vals1)

# finally clipped the image btw 0,255 intensity
pnoisy1 = np.clip(pnoisy1,0,255).astype(np.uint8)

vals2 = len(np.unique(img2_uint8))
vals2 = 2**np.ceil(np.log2(vals2))
pnoisy2 = np.random.poisson(img2_uint8*vals2)/float(vals2)
pnoisy2 = np.clip(pnoisy2,0,255).astype(np.uint8)

#adding images in a list

images = [img_uint8,  gnoisy1, pnoisy1, img2_uint8, gnoisy2, pnoisy2]
titles = [
    'Original1','gaussian noise 1','poisson noise 1',
    'Original2',  'gaussian noise 2','poisson noise 2'
]

#axes 2d np array is created
fig,axes = plt.subplots(2,3,figsize=(15,7))
axes = axes.ravel() # converts 2d np array to 1d, better for iterations

for i in range(6):
    axes[i].imshow(images[i],cmap='gray')
    axes[i].set_title(titles[i])
    axes[i].axis('off')

plt.show()



def process_image(image):
    #as models require it in a fixed size to process any further
    resized_img = cv2.resize(image,(128,128))

    #normalizes btw 0,1, enhances model training
    norm_img = resized_img.astype(np.float32)/255.0

    #convert (h,w) to (h,w,channels) -1 would make 1 channel here which is perfect for grayscale
    norm_img = np.expand_dims(norm_img,-1)

    return norm_img


noisy_images = [gnoisy1, pnoisy1, gnoisy2, pnoisy2]  # noisy images
clean_images = [img_uint8, img_uint8, img2_uint8, img2_uint8]  # corresponding clean images


processed_noisy = [process_image(im) for im in noisy_images]
processed_clean = [process_image(im) for im in clean_images]

X = np.array(processed_noisy)  # model inputs
Y = np.array(processed_clean)  # model targets (labels)

np.save('X.npy', X)
np.save('Y.npy', Y)

print("Images loaded and processed.")
print("Compiling model...")

model = unet_model()
model.compile(optimizer='adam', loss='mean_squared_error')

print("Model compiled.")
print("Starting training...")

history = model.fit(X, Y, epochs=20, batch_size=2, validation_split=0.2,verbose=2)

print("Training finished.")


#cv2.imshow("dicom image 1 uint8",img_uint8)
#cv2.imshow('dicom image 2 uint8',img2_uint8)
#cv2.waitKey(0)

test1 = process_image(gnoisy1)
test2 = process_image(pnoisy1)
test3 = process_image(gnoisy2)
test4 = process_image(pnoisy2)

test_images = [test1, test2, test3, test4]
test_images_names = ['gaussian noise image 1', 'poisson noise image 1', 'gaussian noise image 2', 'poisson noise image 2']

# Predict
denoised_outputs = [model.predict(np.expand_dims(img, axis=0))[0,:,:,0] for img in test_images]

clean_processed = [process_image(img) for img in clean_images]
clean_processed = [img[:, :, 0] for img in clean_processed]
noisy_displayed = [img[:, :, 0] for img in test_images]


titles = ['clean','noisy','denoised']
fig, axes = plt.subplots(4,3,figsize=(12,12))

for i in range(4):
    axes[i, 0].imshow(clean_processed[i], cmap='gray')
    axes[i, 0].set_title(f'{titles[0]} {i + 1}')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(noisy_displayed[i], cmap='gray')
    axes[i, 1].set_title(f'{titles[1]} {i + 1}')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(denoised_outputs[i], cmap='gray')
    axes[i, 2].set_title(f'{titles[2]} {i + 1}')
    axes[i, 2].axis('off')


mse_values = []

for i in range(4):
    # Flatten both clean and predicted (denoised) images to 1D arrays
    mse = mean_squared_error(clean_processed[i].flatten(), denoised_outputs[i].flatten())
    mse_values.append(mse)
    print(f"MSE for Image {i+1}: {mse:.4f}")

ssim_values = []

for i in range(4):
    ssim_val = ssim(clean_processed[i], denoised_outputs[i], data_range=1.0)  # Because values are between 0 and 1
    ssim_values.append(ssim_val)
    print(f"SSIM for Image {i+1}: {ssim_val:.4f}")


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # perfect match
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

for i in range(4):
    psnr_val = psnr(clean_processed[i], denoised_outputs[i])
    print(f"PSNR for Image {i+1}: {psnr_val:.2f} dB")


plt.show()
