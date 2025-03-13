# 다운스케일링: 초반 conv 레이어: 에지, 텍스처 같은 저수준(디테일) 피처 추출, 후반 conv 레이어: 얼굴의 위치, 전체적 형상(추상적, 고차원) 피처 추출
# 그걸 중간중간에 계속 저장해두고, 업스케일링할 때 합쳐서 transpose convolution/upsampling의 예측을 도움
# 업스케일링: 원본 사이즈로 만들어야 분할마스크를 만들고 분류가 가능하니까 초반에는 추상적인 걸 가지고 -> 후반으로 가면서 아까 다운스케일링할때 저장해둔 피처들 가지고 복원해나감
# skip connection을 통해 다운스케일링할 때 정보를 가지고 복원(뭘 conv했길래 이게 나왔을까를 예측)

import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# 경로 정의
image_dir = "C:\\Users\\user\\Desktop\\Task01_BrainTumour\\imagesTr"
label_dir = "C:\\Users\\user\\Desktop\\Task01_BrainTumour\\labelsTr"

transform = transforms.Compose([
    transforms.Resize((128, 128)),            # 이미지를 128x128로 리사이즈
    transforms.ToTensor(),                   # 이미지를 Tensor로 변환
    transforms.Normalize((0.5,), (0.5,)),    # 픽셀 값을 정규화 (-1 ~ 1 범위)
])


# 그럼 부모 클래스의 __init__이나 __getitem__등이 이미 있을 때
# 그 대상을 지정해서 그대로 쓰고 싶으면 super를 쓰는 거고
# 여기서는 그대로 쓰는 건 없고 거의 다 override하기 때문에 그냥 상속만 받은 거
# DataLoader 생성
class NiftiDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform):
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)  # 전체 데이터 개수 반환

    def __getitem__(self, idx):
        # NIfTI 파일 로드
        image_path = os.path.join(self.image_dir,self.image_files[idx])
        label_path = os.path.join(self.label_dir,self.label_files[idx])

        image_data = nib.load(image_path).get_fdata()[:,:,75]
        label_data = nib.load(label_path).get_fdata()[:,:,75]
        
        # 데이터 전처리 (옵션)
        if self.transform:
            image_data = self.transform(image_data)
            label_data = self.transform(label_data)

        # 여기선 레이블 없이 이미지만 반환
        return torch.tensor(image_data, dtype=torch.float32)
    

train_dataset = NiftiDataset(image_dir=image_dir, label_dir=label_dir, transform=transform)
test_dataset = NiftiDataset(image_dir=image_dir, label_dir=label_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 이미지 파일 경로
image_path = "C:\\Users\\user\\Desktop\\Task01_BrainTumour\\imagesTr\\BRATS_001.nii.gz"
label_path = "C:\\Users\\user\\Desktop\\Task01_BrainTumour\\labelsTr\\BRATS_001.nii.gz"
# 이미지 불러오기
img = nib.load(image_path) #nii.gz확장자는 nibabel로 해야함
label = nib.load(label_path)
image_data = img.get_fdata()  # 3D NumPy 배열로 변환 [X,Y,Z,C] 형태 (240, 240, 155, 4)
label_data = label.get_fdata()
print(image_data[:,:,75,:].shape)  # (x, y, z) 형태로 이미지 크기 출력
print(label_data[:,:,75].shape)
fig, axes = plt.subplots(10, 4, figsize=(40, 80))
for x in range(65,75):
    for i in range(4):
        axes[x-65, i].imshow(label_data[:, :, x], cmap='gray')  # 각 채널을 시각화
        #axes[x-60, i].set_title(f'Slice {x}')
        axes[x-65, i].axis('off')
#plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()

class UNet(nn.Module): # 부모클래스는 nn.Module, unet은 nn.Module로부터 무언가를 상속받을 것임 
    def __init__(self, input_channels=3, output_channels=1):
        super(UNet, self).__init__()

        # Encoder: Downsampling path
        self.enc1 = self.double_conv(input_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)
        self.enc5 = self.double_conv(512, 1024)

        # Decoder: Upsampling path
        self.upconv6 = self.upconv_block(1024, 512)
        self.upconv7 = self.upconv_block(512, 256)
        self.upconv8 = self.upconv_block(256, 128)
        self.upconv9 = self.upconv_block(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # Decoder path with skip connections
        up6 = self.upconv6(enc5)
        up6 = torch.cat([up6, enc4], dim=1)
        up7 = self.upconv7(up6)
        up7 = torch.cat([up7, enc3], dim=1)
        up8 = self.upconv8(up7)
        up8 = torch.cat([up8, enc2], dim=1)
        up9 = self.upconv9(up8)
        up9 = torch.cat([up9, enc1], dim=1)

        # Final output layer
        output = self.final_conv(up9)

        return output

# 모델 생성
model = UNet(input_channels=3, output_channels=1)

# 모델 요약 출력 (PyTorch에서는 summary가 기본 제공되지 않으므로, torchsummary나 다른 라이브러리를 사용)
print(model)
