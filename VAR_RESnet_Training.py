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

transforms1 = transforms.Compose([  # 일반적으로 이미지는 (H,W,C)의 형식인데(numpy&PIL) 텐서로 바꿈(C,H,W)
    transforms.RandomHorizontalFlip(p = 0.4),
    transforms.RandomRotation(15,expand=False,fill=128),
    #transforms.Pad(20, fill=128,padding_mode="edge"), # 이건 안하는게...
    transforms.Grayscale(),
    #transforms.Resize((128, 128), interpolation= transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Normalize((0.5),(0.5)),
])

# original_transform = transforms.Compose([
#     transforms.Resize((128, 128), interpolation= transforms.InterpolationMode.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5),(0.5)),
# ])

transforms_test = transforms.Compose([  # 일반적으로 이미지는 (H,W,C)의 형식인데(numpy&PIL) 텐서로 바꿈(C,H,W)
    #transforms.Grayscale(),
    #transforms.Resize((128, 128), interpolation= transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    
    
class myres(nn.Module):
    def __init__(self,input_channels, output_channels, classify_channels, stride = 1):
        super().__init__()
        # GAP사용: 예를 들어 8x8x64의 맵이 입력되었다면, 64개중 1채널의 8x8 배열이 있을 건데, 그 전체를 평균내서 1개의 값으로 바꿀거임 그렇게 64개의 채널에 대해 수행하면,
        # 총 64개의 값으로 이루어진 1차원 배열이 나옴. 이걸 fc_layer에 넘겨줄거임
        self.stride = stride
        self.input_channels =input_channels
        self.output_channels = output_channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.block1 = self.block_iden(32)
        self.block2 = self.block_iden(64)
        self.block3 = self.block_iden(128)
        self.block4 = self.block_iden(256)
        self.downsampling1 = self.downsampling(32,64,stride=2)
        self.downsampling2 = self.downsampling(64,128,stride=2)
        self.downsampling3 = self.downsampling(128,256,stride=2)
        self.initial_conv = nn.Conv2d(1, 32,kernel_size=3,stride=1,padding=1)   
        self.fc_layer = nn.Sequential(
            nn.Linear(256,128), # [64,512*16*16] -> [64,200]
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(128,10),
        )       
        
    def downsampling(self, input_channels, output_channels,stride):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride).to(device),
            nn.BatchNorm2d(output_channels).to(device)
        )
    def shortcut(self, x, input_channels, output_channels, stride):
        """Identity Mapping or 1×1 Conv"""
        if stride != 1 or input_channels != output_channels:
            return self.downsampling(input_channels, output_channels, stride)(x).to(device)
        return x  # Identity mapping

    def block_iden(self, channel):
        return nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size = 3,stride = 1, padding = 1).to(device),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(channel, channel, kernel_size = 3, stride = 1, padding=1).to(device),
            nn.BatchNorm2d(channel),
            nn.Dropout(0.5)
        )
        
    def forward(self,x):
        x= self.initial_conv(x) # 64, 1, 32, 32 -> 64, 32, 32, 32
        #print(f'x.shape={x.shape}')
        enc1 = self.block1(x) # 64, 32, 32, 32
        enc1 += self.shortcut(x,32,32,1) # 64, 32, 32, 32
        enc1_re = nn.ReLU()(enc1) # 64, 32, 32, 32
        
        # enc11 = self.block1(enc1_re)
        # enc11 += self.shortcut(enc1,32,32,1)
        # enc11_re = nn.ReLU()(enc11) # 64, 32, 32, 32
        
        
        
        enc2 = self.downsampling1(enc1_re) # 64, 32, 32, 32 -> 64, 64, 16, 16
        #print(f'enc2.shape={enc2.shape}')
        shortcut_x2 = self.shortcut(enc1,32,64,2) 
        enc2 = self.block2(enc2) 
        enc2 += shortcut_x2 
        enc2_re = nn.ReLU()(enc2) # 64, 64, 16, 16
        
        enc22 = self.block2(enc2_re)
        enc22 += enc2 # 64, 64, 16, 16
        enc22_re = nn.ReLU()(enc22)
        
        
        
        enc3 = self.downsampling2(enc22_re) # 64, 64, 16, 16  ->  64, 128, 8, 8
        shortcut_x3 = self.shortcut(enc22,64,128,2) # 64, 64, 16, 16  ->  64, 128, 8, 8 
        enc3 = self.block3(enc3)
        enc3 += shortcut_x3
        enc3_re = nn.ReLU()(enc3)
        
        enc33 = self.block3(enc3_re)
        enc33 += enc3
        enc33_re = nn.ReLU()(enc33)
        
        
        enc4 = self.downsampling3(enc33_re) # 64, 128, 8, 8   ->  64, 256, 4, 4 
        shortcut_x4 = self.shortcut(enc33,128,256,2)
        enc4 = self.block4(enc4)
        enc4 += shortcut_x4
        enc4_re = nn.ReLU()(enc4)
        
        enc44 = self.block4(enc4_re)
        enc44 += enc4
        enc44_re = nn.ReLU()(enc44)
        
        fin = self.gap(enc44_re)  # 64, 256, 1, 1
        fin = fin.view(fin.size(0),-1)  # 64, 256
        classification = self.fc_layer(fin)
        
        return classification
        
        
train_dataset = mydataset(trainx_path, trainy_path,transforms1) # ,original_transform=original_transform 추가 가능
train_loader = DataLoader(train_dataset, shuffle=True,batch_size=64)
valid_dataset = mydataset(validx_path,validy_path,transforms1) # ,original_transform=original_transform 추가 가능
valid_loader = DataLoader(valid_dataset,shuffle=True,batch_size=64)

epoch_num = 50

model = myres(1,1,10).to(device)
#optimizer = torch.optim.Adam(model.parameters(),lr= 0.0002,weight_decay=1e-3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)


#scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.96) 
criterion = torch.nn.CrossEntropyLoss().to(device)
        
trainloss_list = []
valid_loss_list = []
valid_accu_list = []

for epoch in range(epoch_num):
    model.train()
    running_loss = 0.0
    for images, labels, in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        classification = model(images)
        loss = criterion(classification,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{epoch_num}], Loss: {running_loss/len(train_loader)}')
    trainloss_list.append(running_loss / len(train_loader))
    #scheduler.step()
    
    # validate
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            classification = model(images)
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
        
        
        
torch.save(model.state_dict(), "C:\\Users\\김해창\\Desktop\\cifar-10-batches-py\\model_gray_resnet.pth")    
    
    
plt.plot(trainloss_list, label="Training Loss", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()