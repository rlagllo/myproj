import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from PIL import Image
import cv2
import pytorch_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainx_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\train_x.npz'
trainy_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\train_y.npz'

validx_path =r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\valid_x.npz'
validy_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\valid_y.npz'

testx_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\test_x.npz'
testy_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\test_y.npz'

# train_data = np.load(trainx_path)
# train_label = np.load(trainy_path)

# print(train_data) # np.load를 하면 딕셔너리와 유사한 형태임
# print(train_label)

# valid_label = np.load(validy_path)
# print(valid_label)

transforms = transforms.Compose([  # 일반적으로 이미지는 (H,W,C)의 형식인데(numpy&PIL) 텐서로 바꿈(C,H,W)
    transforms.Grayscale(num_output_channels= 1), # 흑백으로 바꾸기
    transforms.Resize((64, 64), interpolation= transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # 컬러일때 
    transforms.Normalize((0.5),(0.5)) # 흑백일 떄
]) 
 # 이미지는 32x32 사이즈임
# gray scale dataset
# CNN model
class mydataset(Dataset):
    def __init__(self, data_path,label_path, transform = transforms):
        data = np.load(data_path)
        labels = np.load(label_path)
        
        if "train_x" in data and "train_y" in labels:
            self.data = data["train_x"]
            #self.labels = labels["train_y"]
        elif "valid_x" in data and "valid_y" in labels:
            self.data = data["valid_x"]
            #self.labels = labels["valid_y"]
        elif "test_x" in data and "test_y" in labels:
            self.data = data["test_x"]
            #self.labels = labels["test_y"]
        self.transform = transform
        
    def __len__(self): # dataloader는 __len__함수를 통해 데이터 길이를 확인하고, 그에 따라 idx리스트를 만들어냄
        return (len(self.data))
    
    def __getitem__(self,idx): # dataloader가 데이터를 요청할 때 자동으로 호출됨 idx변수는 dataloader가 만든거로 줌
        x = self.data[idx]
        #y = self.labels[idx]
        
        # 정의해놓은 transform의 내용을 보면, torchvision의 것을 쓰고있음. 지금 transforms함수가 torchvision의 것인데, torchvision은 지금 현재 데이터 형식에는 안맞아서, 지원이 되는 PIL형식으로 바꾸는 것
        x = Image.fromarray(np.transpose(x, (1, 2, 0)).astype(np.uint8)) 
        
        if self.transform:
            x = self.transform(x)
        
        noise_std = 0.05
        noise = torch.randn_like(x) * noise_std
        x_noised = x + noise
        x_noised = torch.clamp(x_noised,0,1)
        
        
        return x_noised, x  # 노이즈, 원본
    
train_dataset = mydataset(trainx_path, trainy_path)
train_loader = DataLoader(train_dataset, shuffle=True,batch_size=64)
valid_dataset = mydataset(validx_path,validy_path)
valid_loader = DataLoader(valid_dataset,shuffle=True,batch_size=64)


# sample_image, sample_label = next(iter(train_loader))  # 배치에서 샘플 1개 가져오기
# print("sample image shape: ",sample_image.shape)
# #sample_image = sample_image[1].permute(1, 2, 0).cpu().numpy()  # (C, H, W) → (H, W, C)
# sample_image = sample_image[1].squeeze(0).cpu().numpy()   # 흑백이니까 H,W          
# print("sample image squeezed shape: ", sample_image.shape)             
# plt.imshow(sample_image, cmap="gray")
# plt.title(f"Label: {sample_label[0].item()}")
# plt.show()

# for image, labels in train_loader: # 지금 train_loader안에 있는 image, labels는 pytorch tensor
    
#     img = image[0].permute(1,2,0).numpy() # 여기서 permute를 하면서 텐서가 넘파이로 변환됨
#     img = (img* 0.5) + 0.5
    
#     plt.imshow(img)
#     plt.title(f"Label: {labels[0].item()}")
#     plt.axis("off")
#     plt.show()
#     print("Batch images shape:", image.shape)  # (64, 3, 64, 64)
#     print("Batch labels shape:", labels.shape)  # (64,)
    
    
    # 아이디어 ---> 입력은 64,64의 화질구지     /     출력은 224,224의 복원된 이미지        +       bottleneck에서는 레이블과 위치 도출
    
class unet(nn.Module):
    def __init__(self,input_channels,output_channels): # 입력: RGB 3 출력: RGB 3 판단: 10(레이블)
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
        self.conv_up3 = self.conv2d(512,256)
        self.conv_up2 = self.conv2d(256,128)
        self.conv_up1 = self.conv2d(128,64)

        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)
        
        #self.fc_layer = nn.Linear(512*8*8,classify_channel)
        

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride =1 , padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride =1 , padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def conv2d(self,in_channels,out_channels): # concat해서 2배가 된 채널을 원래거의 절반까지 줄임
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )   
    


    # def upconv_block(self,in_channels,out_channels):
    #     return nn.Sequential(
    #         nn.ConvTranspose2d(in_channels,out_channels,kernel_size=3,stride=2, padding = 1, output_padding = 1).to(device),
    #         nn.BatchNorm2d(out_channels),
    #         nn.ReLU(inplace = True),

    #     )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc11 = self.pool(enc1)
        
        enc2 = self.enc2(enc11)
        enc22 = self.pool(enc2)

        enc3 = self.enc3(enc22)
        enc33 = self.pool(enc3)

        enc4 = self.enc4(enc33)
        #enc44 = self.pool(enc4)
        # print(f'x.shape={x.shape}') # x.shape = torch.Size([64, 1, 64, 64])
        # print(f'enc1.shape={enc1.shape}') # enc1.shape = torch.Size([64, 64, 32, 32])
        # print(f'enc2.shape={enc2.shape}') # enc2.shape=torch.Size([64, 128, 32, 32])
        # print(f'enc3.shape={enc3.shape}') # enc3.shape=torch.Size([64, 256, 16, 16])
        # print(f'enc4.shape={enc4.shape}') # enc4.shape=torch.Size([64, 512, 8, 8])

        up3 = self.upconv3(enc4) # ConvTranspose2d시행 사이즈 8 -> 16, 채널 512 -> 256
        #print(f'up3.shape={up3.shape}') # up3.shape=torch.Size([64, 256, 16, 16])
        up3 = torch.cat([up3, enc3], dim=1) # 여기서 up3 & enc3 합쳐져서 채널 512개가 됨
        up3 = self.conv_up3(up3) # 512 -> 256
        #print(f'up3.shape={up3.shape}') # up3.shape=torch.Size([64, 256, 16, 16])

        up2 = self.upconv2(up3) # 256->128
        #print(f'up2.shape={up2.shape}') # up2.shape=torch.Size([64, 128, 32, 32])
        up2 = torch.cat([up2, enc2], dim=1) # 128->256
        up2 = self.conv_up2(up2) #256->128
        #print(f'up2.shape={up2.shape}') # up2.shape=torch.Size([64, 128, 32, 32])

        up1 = self.upconv1(up2)
        #print(f'up1.shape={up1.shape}') # up1.shape=torch.Size([64, 64, 64, 64])
        up1 = torch.cat([up1, enc1], dim=1)
        up1 = self.conv_up1(up1)
        #print(f'up1_1.shape={up1.shape}') # up1.shape=torch.Size([64, 64, 64, 64])
        #print(f"final_conv(up1).shape = {self.final_conv(up1).shape}") # final_conv(up1).shape = torch.Size([64, 1, 64, 64])
        return self.final_conv(up1)    # restored : 64,3,128,128     classification : 64,10
    
epoch_num = 10


# def gaussian_window(size, sigma):
#     coords = torch.arange(size, dtype=torch.float32) - size // 2
#     g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
#     g /= g.sum()
#     return g[:, None] * g[None, :]

# def create_window(window_size, channel):
#     _1D_window = gaussian_window(window_size, 1.5).unsqueeze(0).unsqueeze(0)
#     window = _1D_window.expand(channel, 1, window_size, window_size).contiguous()
#     return window

# def ssim(img1, img2, window_size=11, size_average=True):
#     channel = img1.size(1)
#     window = create_window(window_size, channel).to(img1.device)

#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
#     return ssim_map.mean() if size_average else ssim_map

# nn.Module을 상속받는다 -> 미분파라미터가 설정된다. to(device) 가능해진다. torch.tensor 연산을 수행하면, 연산 그래프로 연결돼서 .backward()호출이 가능해진다.
# class RMSE_SSIM(nn.Module):
#     def __init__(self, alpha = 0.7):
#         super().__init__()
#         self.alpha = alpha
#         # self.ssimloss = pytorch_ssim.SSIM()
#         self.l2_loss = nn.MSELoss()
        
#     def forward(self,pred,target):
#         ssim_loss = 1 - ssim(pred,target)
#         l2_loss = self.l2_loss(pred,target)
#         return self.alpha * ssim_loss + (1-self.alpha) * torch.sqrt(l2_loss)
    

# model = unet(3,3,10).to(device) # 컬러
model = unet(1, 1).to(device) # 흑백
optimizer = optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=1e-4)
criterion = nn.MSELoss().to(device)
#loss_fn = pytorch_ssim.SSIM()

trainloss_list = []
valid_loss_list = []
#valid_accu_list = []
    
if __name__ == "__main__":
    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0
        for noised, original in train_loader:
            noised, original = noised.to(device), original.to(device)
            optimizer.zero_grad()
            restored = model(noised) 
            loss = criterion(restored,original)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epoch_num}], Loss: {running_loss/len(train_loader)}')
        trainloss_list.append(running_loss / len(train_loader))
        # VALIDATING
        model.eval()
        valid_loss = 0.0
        correct = 0 # 맞춘 문제 수 
        total = 0 # 총 문제 수
        
        with torch.no_grad():
            for noised, original in valid_loader:
                noised, original = noised.to(device), original.to(device)
                restored = model(noised)
                loss = criterion(restored,original)
                #ssim_loss = loss_fn(restored, images)
                valid_loss += loss.item()
                
                
            avg_valid_loss = valid_loss / len(valid_loader)
            valid_loss_list.append(avg_valid_loss)
            
            #valid_accu = 100 * correct / total
            #valid_accu_list.append(valid_accu)

            print(f"Epoch [{epoch+1}/{epoch_num}] | "
            f"Train Loss: {running_loss:.4f} | "
            f"Valid Loss: {avg_valid_loss:.4f} | ")
            #f"Valid Accuracy: {valid_accu:.2f}%")
            # restored_image = restored[0].cpu().permute(1,2,0).numpy()
            # original = original[0].cpu().permute(1,2,0).numpy()
            # noised_sh = noised[0].cpu().permute(1,2,0).numpy()
            
            # fig,axes = plt.subplots(1,3,figsize= (12,4))
            
            # axes[0].imshow(noised_sh, cmap = 'gray')
            # axes[0].set_title(f"noised image")
            # axes[0].axis("off")
            
            # axes[1].imshow(original, cmap = 'gray')
            # axes[1].set_title(f"Original Image")
            # axes[1].axis("off")

            # # 복원된 이미지
            # axes[2].imshow(restored_image, cmap = 'gray')
            # axes[2].set_title("Restored Image")
            # axes[2].axis("off")
            
            # plt.tight_layout()
            # plt.show()
            
        
        
    torch.save(model.state_dict(), "C:\\Users\\김해창\\Desktop\\cifar-10-batches-py\\model_gray_unetrestoration.pth")       
    plt.plot(trainloss_list, label="Training Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()