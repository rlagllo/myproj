
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from PIL import Image
import os
import numpy as np
import glob
import gc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    
    # def rgb_to_class(self, labelimg):
    #     label_class = np.zeros((labelimg.shape[0], labelimg.shape[1]), dtype=np.int64)
    #     for i, color in enumerate(colors):
    #         matches = np.all(labelimg == color, axis=-1)
    #         label_class[matches] = i
    #     return label_class  # (H, W) 형태의 클래스 번호 배열

class mydataset(Dataset):
    def __init__(self, image_path, label_path, transform=None):
        self.image_files = sorted(glob.glob(os.path.join(image_path, '*')))
        self.label_files = sorted(glob.glob(os.path.join(label_path, '*')))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.load(self.image_files[idx])
        label = np.load(self.label_files[idx])

        tensored_image = torch.tensor(image).permute(2,0,1).float()
        tensored_label = torch.tensor(label).long()
        
        return tensored_image, tensored_label

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
        #self.upconv = self.upconv_block(64,32)
        self.conv_up3 = self.conv2d(512,256)
        self.conv_up2 = self.conv2d(256,128)
        self.conv_up1 = self.conv2d(128,64)

        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

        

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride =1 , padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride =1 , padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  #up3.shape=torch.Size([8, 512, 32, 32])
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # up3.shape=torch.Size([8, 256, 32, 32])
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.2)
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), # up3.shape=torch.Size([8, 256, 32, 32])
            # nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )   
    
    def forward(self, x):
        
        enc1 = self.enc1(x) # [64, 64, 128, 128]
        enc11 = self.pool(enc1)
       # print(f'enc1.shape={enc1.shape}') # enc1.shape = torch.Size([64, 64, 256, 256])
        enc2 = self.enc2(enc11)
        enc22 = self.pool(enc2)

        enc3 = self.enc3(enc22)
        enc33 = self.pool(enc3)

        enc4 = self.enc4(enc33)
        #enc44 = self.pool(enc4)
        #print(f'x.shape={x.shape}') # x.shape = torch.Size([64, 3, 256, 256])
        
        # print(f'enc2.shape={enc2.shape}') # enc2.shape=torch.Size([64, 128, 128, 128])
        # print(f'enc3.shape={enc3.shape}') # enc3.shape=torch.Size([64, 256, 64, 64])
        # print(f'enc4.shape={enc4.shape}') # enc4.shape=torch.Size([64, 512, 32, 32])

        up3 = self.upconv3(enc4) # ConvTranspose2d시행 사이즈 8 -> 16, 채널 512 -> 256    up3.shape=torch.Size([8, 256, 32, 32])
        up3 = torch.cat([up3, enc3], dim=1) # 여기서 up3 & enc3 합쳐져서 채널 512개가 됨
        up3 = self.conv_up3(up3) # 512 -> 256

        up2 = self.upconv2(up3) # 256->128
        up2 = torch.cat([up2, enc2], dim=1) # 128->256
        up2 = self.conv_up2(up2) #256->128
       
        up1 = self.upconv1(up2)
        up1 = torch.cat([up1, enc1], dim=1)
        up1 = self.conv_up1(up1)
        #up1 = self.upconv(up1)
        # print(f'up3.shape={up3.shape}') # up3.shape=torch.Size([64, 256, 8, 8])
        # print(f'up2.shape={up2.shape}') # up2.shape=torch.Size([8, 128, 128, 128])
        # print(f'up1.shape={up1.shape}') # up1.shape=torch.Size([8, 64, 256, 256])
        # print(f'up1_1.shape={up1.shape}') # up1.shape=torch.Size([64, 64, 256, 256])
        #print(f"final_conv(up1).shape = {self.final_conv(up1).shape}") # final_conv(up1).shape = torch.Size([64, 29, 512, 512])
        return self.final_conv(up1)    
    
if __name__ == "__main__":
    image_folder_path = "C:\\Users\\김해창\\Desktop\\unet\\cityscapes_data\\processed_train"
    label_foler_path = "C:\\Users\\김해창\\Desktop\\unet\\cityscapes_data\\preprocessed_label"        
    epoch_num = 40

    # nn.Module을 상속받는다 -> 미분파라미터가 설정된다. to(device) 가능해진다. torch.tensor 연산을 수행하면, 연산 그래프로 연결돼서 .backward()호출이 가능해진다.

    dataset = mydataset(image_folder_path, label_foler_path)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4) 

    # images, labelimgs = next(iter(dataset))
    # print("Images shape:", images.shape)   # 예: torch.Size([batch_size, 3, 256, 256])
    # print("Label images shape:", labelimgs.shape)

    # images = images.permute(1,2,0)
    # images = images/255.0
    # plt.imshow(images)
    # plt.show()
    #dataloader = SimpleDataLoader(image_folder_path, label_foler_path, batch_size=16)

    class_to_color = {0: (0, 0, 0), 1: (111, 74, 0), 2: (81, 0, 81), 3: (128, 64, 128), 4: (244, 35, 232), 
                    5: (250, 170, 160), 6: (230, 150, 140), 7: (70, 70, 70), 8: (102, 102, 156), 9: (190, 153, 153), 
                    10: (180, 165, 180), 11: (150, 100, 100), 12: (150, 120, 90), 13: (153, 153, 153), 
                    14: (250, 170, 30), 15: (220, 220, 0), 16: (107, 142, 35), 17: (152, 251, 152), 18: (70, 130, 180),
                    19: (220, 20, 60), 20: (255, 0, 0), 21: (0, 0, 142), 22: (0, 0, 70), 23: (0, 60, 100), 24: (0, 0, 90), 
                    25: (0, 0, 110), 26: (0, 80, 100), 27: (0, 0, 230), 28: (119, 11, 32)}

    model = unet(3,29).to(device) # 컬러
    # optimizer = optim.RMSprop(model.parameters(), lr=0.001,
    #                    momentum=0.92, weight_decay=1e-7)
    optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999),eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    #criterion = RMSE_SSIM().to(device)
    #loss_fn = pytorch_ssim.SSIM()

    trainloss_list = []
    valid_loss_list = []
    #valid_accu_list = []    9:14분  
        
    if __name__ == "__main__":
        for epoch in range(epoch_num):
            model.train()
            running_loss = 0.0
            for images, labelimgs in dataloader:
                #print(images.shape) # 64, 3, 256, 256
                #print(labelimgs.shape) # 64, 256, 256
                images, labelimgs = images.to(device), labelimgs.to(device)
                optimizer.zero_grad()
                segmented = model(images) 
                loss = criterion(segmented,labelimgs)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            print(f'Epoch [{epoch+1}/{epoch_num}], Loss: {running_loss/len(dataloader)}')
            trainloss_list.append(running_loss / len(dataloader))
            # segmented_image = torch.argmax(segmented,dim=1)  # segmented.shape = (batch_size, num_classes, H, W)이고, argmax로 dim = 1을 하면 num_classes채널에서 가장 큰 애의 인덱스를 불러옴
            # #original = images[0].cpu().permute(1,2,0)    # N H W -> H W C로
            # labelimg_sh = segmented_image[0].cpu()
            # ans_label = labelimgs[0].reshape(256,256,3)
            # rgb_img = np.array([class_to_color[int(val)] for val in labelimg_sh.flatten()])
            # rgb_img = rgb_img.reshape(256, 256, 3)
            # original_image = images[0].permute(1,2,0).reshape(256,256,3).cpu().numpy()
            # original_image = original_image / 255.0
            
            # fig, ax = plt.subplots(1,3, figsize = (10,5))
            # ax[0].imshow(original_image)
            # ax[0].axis('off')  
            # ax[1].imshow(ans_label)
            # ax[2].imshow(rgb_img)
            # ax[2].axis('off')  
            # plt.show()
            del images, labelimgs, segmented, loss
            torch.cuda.empty_cache()  # GPU 캐시 해제
            gc.collect()
            
            
            # # VALIDATING
            # model.eval()
            # valid_loss = 0.0
            # correct = 0 # 맞춘 문제 수 
            # total = 0 # 총 문제 수
            
            # with torch.no_grad():
            #     for images, labelimg in dataloader:
            #         images, labelimg = images.to(device), labelimg.to(device)
            #         segmented = model(images)
            #         loss = criterion(segmented,labelimg)
            #         #ssim_loss = loss_fn(restored, images)
            #         valid_loss += loss.item()
                    
                    
            #     avg_valid_loss = valid_loss / len(dataloader)
            #     valid_loss_list.append(avg_valid_loss)
                
            #     #valid_accu = 100 * correct / total
            #     #valid_accu_list.append(valid_accu)

            #     print(f"Epoch [{epoch+1}/{epoch_num}] | "
            #     f"Train Loss: {running_loss:.4f} | "
            #     f"Valid Loss: {avg_valid_loss:.4f} | ")
            #     #f"Valid Accuracy: {valid_accu:.2f}%")
            #     retored_image = segmented[0].cpu().permute(1,2,0).numpy()
            #     original = images[0].cpu().permute(1,2,0).numpy()
                
            #     fig,axes = plt.subplots(1,2,figsize= (12,4))
            #     axes[0].imshow(original, cmap = 'gray')
            #     axes[0].set_title(f"Original Image")
            #     axes[0].axis("off")

            #     # 복원된 이미지
            #     axes[1].imshow(retored_image, cmap = 'gray')
            #     axes[1].set_title("Restored Image")
            #     axes[1].axis("off")
                
            #     plt.tight_layout()
            #     plt.show()
                
            
            
        torch.save(model.state_dict(), "C:\\Users\\김해창\\Desktop\\cifar-10-batches-py\\model_grayV4_notRMS_ADAM.pth")       
        plt.plot(trainloss_list, label="Training Loss", marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
    #class SimpleDataLoader:
        # def __init__(self, image_path, label_path, batch_size=8, shuffle=False):
        #     self.image_path = image_path
        #     self.label_path = label_path
        #     self.batch_size = batch_size
        #     self.shuffle = shuffle
            
        #     # 폴더의 모든 npy 경로 저장
        #     self.images = glob.glob(os.path.join(image_path,'*'))
        #     self.labels = glob.glob(os.path.join(label_path,'*'))
        #     self.num_images = len(self.images)

        #     self.indices = np.arange(self.num_images)
        #     if self.shuffle:
        #         np.random.shuffle(self.indices)
        
        # def __len__(self):
        #     # 총 배치의 개수
        #     return int(np.ceil(self.num_images / self.batch_size))
        
        # def __iter__(self): # for gt, labelimg in dataloder: 구문이 수행되면 바로 __iter__가 호출됨
        #     self.current = 0 # 현재 배치를 가리키는 포인터를 0으로 초기화
        #     return self # 
        
        # def __next__(self): # __iter__ 수행이후에 __next__호출
        #     #print("next")
        #     if self.current >= self.num_images:
        #         raise StopIteration
        #     # 현재 위치(self.current)에서 배치사이즈만큼의 인덱스를 가져옴
        #     batch_indices = self.indices[self.current:self.current + self.batch_size]
        #     # load_image함수를 통해 그 인덱스만큼의 파일을 로드
        #     batch_images = [self.load_image(self.images[i]) for i in batch_indices]
        #     batch_labels = [self.load_image(self.labels[i]) for i in batch_indices]
        #     # 위에서 가져온 배치사이즈 만큼의 이미지들을 합침
        #     batch_images = torch.stack(batch_images, dim =0)
        #     batch_labes = torch.stack(batch_labels, dim=0)

        #     self.current += self.batch_size
        #     return batch_images, batch_labes # 정리: 배치사이즈만큼의 파일들을 load_image함수를 이용해 열고, 2개의 큰 배열 gts, labelimgs로 반환
        
        # def load_image(self, path):
        #     #print("load_image")
        #     file = np.load(path) # H W C    /     H,W
        #     if file.ndim == 2:
        #         tensored = torch.tensor(file).long()
        #     else:
        #         tensored = torch.tensor(file).permute(2,0,1).float()
        #     return tensored
        
        # 12