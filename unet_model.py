from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

def unet_model(input_size=(64, 64, 1)):
    """
    U-Net: Noisy LL (64x64) → Clean LL (64x64).
    No extra upsampling needed — IDWT handles reconstruction.
    """
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3,3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)   # 32x32

    c2 = Conv2D(64, (3,3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3,3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)   # 16x16

    c3 = Conv2D(128, (3,3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3,3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)   # 8x8

    # Bottleneck
    c4 = Conv2D(256, (3,3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3,3), activation='relu', padding='same')(c4)

    # Decoder
    u5 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c4)  # 16x16
    u5 = concatenate([u5, c3])
    c5 = Conv2D(128, (3,3), activation='relu', padding='same')(u5)
    c5 = Conv2D(128, (3,3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c5)   # 32x32
    u6 = concatenate([u6, c2])
    c6 = Conv2D(64, (3,3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3,3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c6)   # 64x64
    u7 = concatenate([u7, c1])
    c7 = Conv2D(32, (3,3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3,3), activation='relu', padding='same')(c7)

    # Output: clean LL (64x64) — sigmoid keeps it in [0,1]
    outputs = Conv2D(1, (1,1), activation='sigmoid')(c7)

    return Model(inputs=[inputs], outputs=[outputs])