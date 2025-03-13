import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import nibabel as nib
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.chdir("C:\\Users\\김해창\\Desktop\\Task01_BrainTumour")

train_dir = "C:\\Users\\김해창\\Desktop\\Task01_BrainTumour\\imagesTr"
label_dir = "C:\\Users\\김해창\\Desktop\\Task01_BrainTumour\\labelsTr"
test_dir = "C:\\Users\\김해창\\Desktop\\Task01_BrainTumour\\imagesTs"

transform = transforms.Compose([
    transforms.Resize((512, 512)),            # 이미지를 128x128로 리사이즈
    transforms.ToTensor(),                   # 이미지를 Tensor로 변환
    transforms.Normalize((0.5,), (0.5,)),    # 픽셀 값을 정규화 (-1 ~ 1 범위), 텐서로 바꾸고 리사이즈, 정규화하니까 [C,H,W]로 나올것임
])
transform2 = transforms.Compose([
    transforms.Resize((512,512),interpolation=Image.NEAREST)
])

class NiftiDataset(Dataset): # init, len, getitem으로 구성
    def __init__(self, image_dir, label_dir, image_transform, label_transform):
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.image_files)  # 전체 데이터 개수 반환

    def __getitem__(self, idx):
        # NIfTI 파일 로드
        image_path = os.path.join(self.image_dir,self.image_files[idx])
        label_path = os.path.join(self.label_dir,self.label_files[idx])

        image_data = nib.load(image_path).get_fdata()[:,:,75,:]
        label_data = nib.load(label_path).get_fdata()[:,:,75]


        image_data = Image.fromarray(image_data.astype(np.uint8))  # uint8 형식으로 변환
        label_data = Image.fromarray(label_data.astype(np.uint8))



        # 데이터 전처리 (옵션)
        if self.image_transform:
            image_data = self.image_transform(image_data)


        if self.label_transform:
            label_data = self.label_transform(label_data)
            label_data = torch.tensor(np.array(label_data), dtype=torch.long)
        # print("Raw label unique values:", np.unique(label_data))  # 원본 데이터 확인
        # 여기선 레이블 없이 이미지만 반환
        return torch.tensor(image_data, dtype=torch.float32).to(device), torch.tensor(label_data, dtype=torch.long).to(device)

train_dataset = NiftiDataset(image_dir=test_dir, label_dir=label_dir, image_transform=transform, label_transform=transform2)
test_dataset = NiftiDataset(image_dir=test_dir, label_dir=label_dir, image_transform=transform, label_transform=transform2)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("train 개수: ",len(train_dataset))


class unet(nn.Module):
    def __init__(self,input_channels,output_channels):
        super().__init__()

        self.enc1 = self.double_conv(input_channels,64)
        self.enc2 = self.double_conv(64,128)
        self.enc3 = self.double_conv(128,256)
        self.enc4 = self.double_conv(256,512)

        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)

        self.upconv3 = self.upconv_block(512,256)
        self.upconv2 = self.upconv_block(256,128)
        self.upconv1 = self.upconv_block(128,64)
        self.upconv = self.upconv_block(64,32)

        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride =1 , padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride =1 , padding=1),
            nn.ReLU(inplace=True)
        )

    def conv2d(self,x,in_channels,out_channels): # concat해서 2배가 된 채널을 원래거의 절반까지 줄임
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1).to(device)
        return conv(x)


    def upconv_block(self,in_channels,out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size=3,stride=2,padding = 1, output_padding = 1).to(device),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        # print(f'x.shape={x.shape}') # x.shape=torch.Size([64, 4, 512, 512])
        enc1 = self.enc1(x)
        enc1 = self.pool(enc1)
        #print(f'enc1.shape={enc1.shape}') # enc1.shape=torch.Size([64, 64, 256, 256])

        enc2 = self.enc2(enc1)
        enc2 = self.pool(enc2)
        #print(f'enc2.shape={enc2.shape}') # enc2.shape=torch.Size([64, 128, 128, 128])

        enc3 = self.enc3(enc2)
        enc3 = self.pool(enc3)
        #print(f'enc3.shape={enc3.shape}') # enc3.shape=torch.Size([64, 256, 64, 64])

        enc4 = self.enc4(enc3)
        enc4 = self.pool(enc4)
        #print(f'enc4.shape={enc4.shape}') # enc4.shape=torch.Size([64, 512, 32, 32])

        up3 = self.upconv3(enc4) # ConvTranspose2d시행 512->256
        up3 = torch.cat([up3, enc3], dim=1) # 여기서 up3 & enc3 합쳐져서 채널 512개가 됨
        up3 = self.conv2d(up3,512,256)
        #print(f'up3.shape={up3.shape}') # up3.shape=torch.Size([64, 256, 64, 64])

        up2 = self.upconv2(up3) # 256->128
        #print(f'up2.shape={up2.shape}') # up2.shape=torch.Size([64, 128, 128, 128])
        up2 = torch.cat([up2, enc2], dim=1) # 128->256
        up2 = self.conv2d(up2,256,128) #256->128
        #print(f'up2.shape={up2.shape}') # up2.shape=torch.Size([64, 128, 128, 128])

        up1 = self.upconv1(up2)
        #print(f'up1.shape={up1.shape}') # up1.shape=torch.Size([64, 64, 256, 256])
        up1 = torch.cat([up1, enc1], dim=1)
        up1 = self.conv2d(up1,128,64)
        up1 = self.upconv(up1)
        # print(f'up1.shape={up1.shape}') # up1.shape=torch.Size([64, 32, 512, 512])
        # print(f"final_conv(up1).shape = {self.final_conv(up1)}")
        return self.final_conv(up1)
if __name__ == "__main__":
    train_loss = []
    num_epochs = 10
    model = unet(input_channels = 4,output_channels =4).to(device)

    #for name, param in model.named_parameters():
        #print(f"{name} is on {param.device}")  # 모델 가중치가 GPU에 있는지 확인

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("훈련시작")
    for epoch in range(num_epochs):
        #print("model: ", model)
        model.train()
        running_loss = 0.0
        for inputs,labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = torch.clamp(labels, min=0).long()
            # print("labels: ", labels)
            # print(f"labels unique: {torch.unique(labels)}")
            # print("After processing: ", torch.unique(labels))
            # print("type(labels): ",type(labels))  # Python 타입 <class 'torch.Tensor'>
            # print("labels.dtype: ",labels.dtype)  # Tensor의 데이터 타입 torch.int64
            # print(f"labels.shape = {labels.shape}") # labels.shape = torch.Size([64, 1, 512, 512]) 512 512 이미지에서 이진
            optimizer.zero_grad()
            outputs = model(inputs)
            #input_image = inputs[0].cpu().numpy()    
            #predictions = torch.argmax(outputs,dim=1) # 각 픽셀에 대해 레이블값 (0,1,2,3)중에 뭔지 할당
            #print(predictions)
            #input_images = inputs[0].cpu().numpy()  # [C, H, W] 형태 (4, 512, 512)
            #predicted_mask = predictions[0].cpu().numpy()  # [H, W] 형태 (512, 512)

            # # ✅ 4개의 채널을 각각 시각화
            # fig, axes = plt.subplots(1, 5, figsize=(20, 5))

            # for i in range(4):  # 4개의 채널
            #     img = input_images[i]  # 채널별 이미지
            #     axes[i].imshow(img, cmap="gray")
            #     axes[i].set_title(f"Input Channel {i+1}")
            #     axes[i].axis("off")

            # # ✅ 예측된 Segmentation Mask 시각화
            # axes[4].imshow(predicted_mask, cmap="jet")  # 예측된 레이블 (0,1,2,3)
            # axes[4].set_title("Predicted Segmentation Mask")
            # axes[4].axis("off")

            # plt.show()
            # print(outputs.dtype)  # outputs 데이터 타입 torch.float32
            # print(outputs.shape) # torch.Size([64, 4, 512, 512])
            loss = criterion(outputs,labels.squeeze(1).long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')
        train_loss.append(running_loss / len(train_loader))

    torch.save(model.state_dict(), "C:\\Users\\김해창\\Desktop\\Task01_BrainTumour\\model.pth")
    plt.plot(train_loss, label="Training Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()