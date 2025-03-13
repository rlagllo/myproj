import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import skimage.transform
from PIL import Image
import cv2
import torch.optim.lr_scheduler as lr_scheduler
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

transforms1 = transforms.Compose([  # 일반적으로 이미지는 (H,W,C)의 형식인데(numpy&PIL) 텐서로 바꿈(C,H,W)
    transforms.RandomHorizontalFlip(p = 0.4),
    transforms.RandomRotation(15,expand=False,fill=128),
    #transforms.Pad(20, fill=128,padding_mode="edge"), # 이건 안하는게...
    transforms.Grayscale(),
    transforms.Resize((128, 128), interpolation= transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    transforms.Normalize((0.5),(0.5)),
    #transforms.Lambda(lambda img: img.convert("RGB"))
])

# original_transform = transforms.Compose([
#     transforms.Resize((128, 128), interpolation= transforms.InterpolationMode.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5),(0.5)),
# ])

transforms_test = transforms.Compose([  # 일반적으로 이미지는 (H,W,C)의 형식인데(numpy&PIL) 텐서로 바꿈(C,H,W)
    transforms.Grayscale(),
    transforms.Resize((128, 128), interpolation= transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    transforms.Normalize((0.5),(0.5)),
    #transforms.Lambda(lambda img: img.convert("RGB"))
])


 # 이미지는 32x32 사이즈임
# gray scale dataset
# CNN model
class mydataset(Dataset):
    def __init__(self, data_path,label_path, transform ): # ,original_transform 추가 가능
        data = np.load(data_path)
        labels = np.load(label_path)
        
        if "train_x" in data and "train_y" in labels:
            self.data = data["train_x"]
            self.labels = labels["train_y"]
        elif "valid_x" in data and "valid_y" in labels:
            self.data = data["valid_x"]
            self.labels = labels["valid_y"]
        elif "test_x" in data and "test_y" in labels:
            self.data = data["test_x"]
            self.labels = labels["test_y"]
        self.transform = transform
        #self.original_transform = original_transform
        
    def __len__(self): # dataloader는 __len__함수를 통해 데이터 길이를 확인하고, 그에 따라 idx리스트를 만들어냄
        return (len(self.data))
    
    def __getitem__(self,idx): # dataloader가 데이터를 요청할 때 자동으로 호출됨 idx변수는 dataloader가 만든거로 줌
        x = self.data[idx]
        y = self.labels[idx]
        
        # 정의해놓은 transform의 내용을 보면, torchvision의 것을 쓰고있음. 지금 transforms함수가 torchvision의 것인데, torchvision은 지금 현재 데이터 형식에는 안맞아서, 지원이 되는 PIL형식으로 바꾸는 것
        x = Image.fromarray(np.transpose(x, (1, 2, 0)).astype(np.uint8)) 
        #original = self.original_transform(x)
        
        
        if self.transform:
            x = self.transform(x)
            
        return x, torch.tensor(y,dtype=torch.long)
        # return original, x, torch.tensor(y,dtype=torch.long)
    





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
    
class mycnn(nn.Module):
    def __init__(self,input_channels,output_channels,classify_channel): # 입력: RGB 3 출력: RGB 3 판단: 10(레이블)
        super().__init__()

        self.enc1 = self.double_conv(input_channels,64)
        self.enc2 = self.double_conv(64,128)
        self.enc3 = self.double_conv(128,256)
        self.enc4 = self.double_conv(256,512) # 이건 필수일 수 있음

        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)

        # self.upconv3 = self.upconv_block(512,256)
        # self.upconv2 = self.upconv_block(256,128)
        # self.upconv1 = self.upconv_block(128,64)
        # self.upconv = self.upconv_block(64,32)

        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=1)
        
        self.fc_layer = nn.Sequential(
            nn.Linear(512*16*16,300), # [64,512*16*16] -> [64,200]
            nn.BatchNorm1d(300),
            nn.ReLU(), # 얘는 파라미터 없음
            nn.Dropout(p = 0.5), # 얘도 없음
            nn.Linear(300,512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10) #[64,100] -> [64,10]
        )       
        
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride =1 , padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride =1 , padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # def upconv_block(self,in_channels,out_channels):
    #     return nn.Sequential(
    #         nn.ConvTranspose2d(in_channels,out_channels,kernel_size=3,stride=2,padding = 1, output_padding = 1).to(device),
    #         nn.ReLU(inplace = True),
    #         nn.Dropout(0.4)
    #     )
    def forward(self, x):
        #print(f'x.shape={x.shape}') # x.shape=torch.Size([64, 3, 128, 128])   흑백: torch.Size([64, 1, 128, 128])
        enc1 = self.enc1(x)
        enc1 = self.pool(enc1)
        #print(f'enc1.shape={enc1.shape}') # enc1.shape=torch.Size([64, 64, 64, 64])   흑백: torch.Size([64, 64, 64, 64])

        enc2 = self.enc2(enc1)
        enc2 = self.pool(enc2)
        #print(f'enc2.shape={enc2.shape}') # enc2.shape=torch.Size([64, 128, 32, 32])

        enc3 = self.enc3(enc2)
        enc3 = self.pool(enc3)
        #print(f'enc3.shape={enc3.shape}') # enc3.shape=torch.Size([64, 256, 16, 16])

        enc4 = self.enc4(enc3)  # conv 레이어 하나 빼고 테스트 torch.Size([64, 512, 16, 16])
        #enc4 = self.pool(enc4)
        #print(f'enc4.shape={enc4.shape}') 

        flatten = enc4.view(enc4.size(0),-1)
        #print(f'flatten.shape=', flatten.shape) # torch.Size([64, 512*16*16])
        
        classification = self.fc_layer(flatten)
        #print(f"\nfeatures.shape: {enc4.shape}")
        # up3 = self.upconv3(enc4) # ConvTranspose2d시행 512->256
        # up3 = torch.cat([up3, enc3], dim=1) # 여기서 up3 & enc3 합쳐져서 채널 512개가 됨
        # up3 = self.conv2d(up3,512,256)
        # #print(f'up3.shape={up3.shape}') # up3.shape=torch.Size([64, 256, 64, 64])

        # up2 = self.upconv2(up3) # 256->128
        # #print(f'up2.shape={up2.shape}') # up2.shape=torch.Size([64, 128, 128, 128])
        # up2 = torch.cat([up2, enc2], dim=1) # 128->256
        # up2 = self.conv2d(up2,256,128) #256->128
        # #print(f'up2.shape={up2.shape}') # up2.shape=torch.Size([64, 128, 128, 128])

        # up1 = self.upconv1(up2)
        # #print(f'up1.shape={up1.shape}') # up1.shape=torch.Size([64, 64, 256, 256])
        # up1 = torch.cat([up1, enc1], dim=1)
        # up1 = self.conv2d(up1,128,64)
        # up1 = self.upconv(up1)
        # # print(f'up1.shape={up1.shape}') # up1.shape=torch.Size([64, 32, 512, 512])
        # # print(f"final_conv(up1).shape = {self.final_conv(up1).shape}")
        return classification, enc4    # restored : 64,3,128,128     classification : 64,10
    
    
if __name__ == "__main__":
    train_dataset = mydataset(trainx_path, trainy_path,transforms1) # ,original_transform=original_transform 추가 가능
    train_loader = DataLoader(train_dataset, shuffle=True,batch_size=64)
    valid_dataset = mydataset(validx_path,validy_path,transforms1) # ,original_transform=original_transform 추가 가능
    valid_loader = DataLoader(valid_dataset,shuffle=True,batch_size=64)
    # # classes =  ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    # sample_image, sample_label = next(iter(train_loader))  # 배치에서 샘플 1개 가져오기 sample_original, 추가가능
    # print(sample_image.shape)
    # #sample_image = sample_image[1].permute(1,2,0).cpu().numpy()
    # #sample_original = sample_original[1].permute(1,2,0).cpu().numpy()
    # sample_image = sample_image[1].squeeze(0).cpu().numpy()  # (C, H, W) → (H, W, C): permute    흑백일 때, squeeze: 1,H,W -> H,W: squeeze
    # fig, ax = plt.subplots(1,2,figsize = (10,5))
    # ax[0].text(0.5, 0.5, str(sample_label[1].item()), fontsize=20, ha='center', va='center')
    # ax[0].set_title("label")
    # ax[0].axis('off')
    # ax[1].imshow(sample_image, cmap = 'gray')
    # ax[1].set_title("gray image")
    # ax[1].axis('off')
    # plt.show()

    
    epoch_num = 50

    model = mycnn(1,1,10).to(device)
    #model = mycnn(3, 3, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr= 0.0002)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.97) 
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    trainloss_list = []
    valid_loss_list = []
    valid_accu_list = []

    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            classification,feature = model(images) 
            loss = criterion(classification,labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epoch_num}], Loss: {running_loss/len(train_loader)}')
        trainloss_list.append(running_loss / len(train_loader))
        scheduler.step()
        
        
        # VALIDATING
        model.eval()
        valid_loss = 0.0
        correct = 0 # 맞춘 문제 수 
        total = 0 # 총 문제 수

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                classification,feature = model(images)
                loss = criterion(classification,labels)
                valid_loss += loss.item()
                
                _,predicted = torch.max(classification,1)
                total += labels.size(0) # 차원에 해당하는 크기를 반환, size(0): 배치 크기(샘플사진개수) size(n) : n번째 차원의 크기
                correct += (predicted == labels).sum().item()
                
            avg_valid_loss = valid_loss / len(valid_loader)
            valid_loss_list.append(avg_valid_loss)
            
            valid_accu = 100 * correct / total
            valid_accu_list.append(valid_accu)

            print(f"Epoch [{epoch+1}/{epoch_num}] | "
            f"Train Loss: {running_loss:.4f} | "
            f"Valid Loss: {avg_valid_loss:.4f} | "
            f"Valid Accuracy: {valid_accu:.2f}%")
            # original = images[0].cpu().squeeze(0).numpy()
            # _, prediction = torch.max(classification[0], dim=0)
            # actual = labels[0].item()
            
            # fig,axes = plt.subplots(1,2,figsize= (12,4))
            # axes[0].imshow(original,cmap = 'gray')
            # axes[0].set_title(f"Original Image (Label: {actual})")
            # axes[0].axis("off")
            # #예측된 클래스 표시
            # axes[1].text(0.5, 0.5, f"Predicted Class: {prediction.item()}", 
            #             fontsize=20, ha="center", va="center")
            # axes[1].set_title("Classification Result")
            # axes[1].axis("off")

            # plt.tight_layout()
            # plt.show()
            
    torch.save(model.state_dict(), "C:\\Users\\김해창\\Desktop\\cifar-10-batches-py\\model_gray_CNN.pth")    
        
        
    plt.plot(trainloss_list, label="Training Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()
    
    
    #Epoch [2/32] | Train Loss: 899.9057 | Valid Loss: 1.3343 | Valid Accuracy: 53.06%