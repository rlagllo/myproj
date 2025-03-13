# 다운스케일링: 초반 conv 레이어: 에지, 텍스처 같은 저수준(디테일) 피처 추출, 후반 conv 레이어: 얼굴의 위치, 전체적 형상(추상적, 고차원) 피처 추출
# 그걸 중간중간에 계속 저장해두고, 업스케일링할 때 합쳐서 transpose convolution/upsampling의 예측을 도움
# 업스케일링: 원본 사이즈로 만들어야 분할마스크를 만들고 분류가 가능하니까 초반에는 추상적인 걸 가지고 -> 후반으로 가면서 아까 다운스케일링할때 저장해둔 피처들 가지고 복원해나감
# skip connection을 통해 다운스케일링할 때 정보를 가지고 복원(뭘 conv했길래 이게 나왔을까를 예측)

import tensorflow as tf
from keras import layers, models

def unet_model(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)
    
    # Encoder: Downsampling path
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder: Upsampling path
    u6 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4], axis=3)  # skip connection
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3], axis=3)  # skip connection
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2], axis=3)  # skip connection
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)  # skip connection
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    # Model
    model = models.Model(inputs, outputs)

    return model

# 모델 생성
model = unet_model()

# 모델 요약 출력
model.summary()