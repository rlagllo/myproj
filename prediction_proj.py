import os
os.chdir('C:\\Users\\김해창\\Desktop\\mesh_extraction_tool_')
import glob
import numpy as np
import datetime
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torch.autograd import Variable
#클래스 객체 생성과 동시에 csv 파일을 depth image로 변환, 그리고 결과 색깔사진도 리사이즈해서 학습,,

def point_cloud_to_image(point_cloud, img_size=(256, 256)):
    # 포인트 클라우드 X, Y 좌표 정규화 최소/최대 범위에 맞게 변환
    x_min, y_min = point_cloud[:, 0].min(), point_cloud[:, 1].min()
    x_max, y_max = point_cloud[:, 0].max(), point_cloud[:, 1].max()
    z_min, z_max = point_cloud[:, 2].min(), point_cloud[:, 2].max()
    # 포인트 클라우드를 투영 X, Y 좌표로 매핑, Z는 깊이
    img = torch.zeros(img_size)  # 초기 이미지 생성
    
    for x, y, z in point_cloud:
        iy = min(int((x - x_min) / (x_max - x_min) * img_size[0]), img_size[0] - 1)
        ix = min(int((y - y_min) / (y_max - y_min) * img_size[1]), img_size[1] - 1)
        img[ix, iy] = max(img[ix, iy], -z)  # 깊이 값 추가
        inverted_z = 2 * ((z - z_min) / (z_max - z_min)) - 1  # z값을 [-1, 1] 범위로 정규화
        img[ix, iy] = max(img[ix, iy], -inverted_z)  # 반전된 깊이 값 추가
    img = 1-img
    return img.unsqueeze(0)  # 채널 추가 (C=1)


class VictorianDataset(Dataset):
    def __init__(self, root):
        # 파일 경로 설정
        self.point_cloud_files = sorted(glob.glob(os.path.join(root,'vertices') + "/*.*"))
        self.color_files = sorted(glob.glob(os.path.join(root, 'resized') + "/*.*"))
        
        # 기본 변환
        self.basic_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # 이미지 데이터 로드 및 통계 계산
        point_clouds = []
        color_imgs = []

        for point_cloud_file in self.point_cloud_files:
            # Point Cloud 로드 
            point_cloud = np.loadtxt(point_cloud_file, delimiter=',', skiprows=1)  # (N, 3) 형태로 로드
            point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float32)  # Tensor로 변환
            point_clouds.append(point_cloud_tensor)
        print("point cloud 변환 완료")
        
        for color_file in self.color_files:
            color_img = Image.open(color_file).convert("RGB")
            color_img = self.basic_transform(color_img)
            #print('color image min, max:', color_img.min(), color_img.max())  #정상 확인, color는 모두 텐서로 넘어가면서 0,1 범위로 정규화됨
            color_imgs.append(color_img)

        point_clouds = torch.stack(point_clouds)   #파일 안의 이미지가 여러개일 것이므로, 옆으로 주루룩 붙여서 배치함
        color_imgs = torch.stack(color_imgs)    #shape torch.Size([8, 3, 256, 256])
        print('Point Cloud shape:', point_clouds.shape)
        print('color shape', color_imgs.shape)        
        
        # Mean과 Std를 직접 입력
        self.gray_mean = [0.9442619681358337]
        self.gray_std = [0.22733019292354584]
        self.color_mean = [0.23110634088516235, 0.4471947252750397, 0.676555871963501]
        self.color_std = [0.3657154440879822, 0.4159409999847412, 0.34875696897506714]
    
        # Normalize를 포함한 최종 변환
        self.color_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.color_mean, std=self.color_std),
        ])

        self.depth_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=self.gray_mean, std=self.gray_std),
        ])
        
    def __getitem__(self, index):

        # Point Cloud 로드 및 변환
        point_cloud = np.loadtxt(self.point_cloud_files[index],delimiter=',',skiprows=1)
        point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float32)
        
        depth_image = point_cloud_to_image(point_cloud_tensor)  # 이미지로 변환
        depth_image = self.depth_transforms(depth_image)
        
        color_file = self.color_files[index % len(self.color_files)]
        color_img = Image.open(color_file).convert("RGB")
        color_img = self.color_transforms(color_img)
        # 디버깅용으로 depth_image 시각화
        depth_image_np = depth_image.squeeze(0).cpu().numpy()  # 배치 차원 제거하고 NumPy로 변환
        plt.imshow(depth_image_np, cmap='gray')  # 흑백 이미지로 시각화
        plt.title("Depth Image")
        plt.colorbar()  # 색상 바 표시
        plt.show()
        return {"A": depth_image, "B": color_img}

    def __len__(self):
        return len(self.point_cloud_files)

root = ''
print('root')
batch_size = 1
img_height = 256
img_width = 256

dataset = VictorianDataset(root)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

gray_mean = dataset.gray_mean
gray_std = dataset.gray_std
color_mean = dataset.color_mean
color_std = dataset.color_std

print('gray mean', gray_mean)
print('gray std', gray_std)
print('color mean', color_mean)
print('color std', color_std)

def reNormalize(img, mean, std):
    img = img.numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = img.clip(0, 1)
    return img

fig = plt.figure(figsize=(10, 5))
rows = 1
cols = 2

for X in train_loader:
    print(X['A'].shape, X['B'].shape)

    # Extract only the first channel of the gray image 흑백 이미지 시각화
    gray_img_first_channel = reNormalize(X["A"][0], gray_mean, gray_std)  # Extract 1st channel
    print(gray_img_first_channel.min(), gray_img_first_channel.max(),'복원 흑백 수치 확인')

    # Plot gray image 
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(gray_img_first_channel, cmap='gray')  # Use cmap='gray' for grayscale visualization
    ax1.set_title('gray img (1st channel)')
    print("여기?")

    # Plot color image
    color_img = reNormalize(X["B"][0], color_mean, color_std)
    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(color_img)
    ax2.set_title('color img')

    #plt.show()
    plt.close(fig)
    #plt.close('all')
    break

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# U-NET 생성

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

GeneratorUNet().apply(weights_init_normal)

Discriminator().apply(weights_init_normal)

n_epochs = 1200
dataset_name = "Victorian400"
lr = 0.0002
b1 = 0.5                    # adam: decay of first order momentum of gradient
b2 = 0.999                  # adam: decay of first order momentum of gradient
decay_epoch = 1000           # epoch from which to start lr decay
#n_cpu = 8                   # number of cpu threads to use during batch generation
channels = 3                # number of image channels
checkpoint_interval = 20    # interval between model checkpoints

os.makedirs("images/%s/val" % dataset_name, exist_ok=True)
os.makedirs("images/%s/test" % dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % dataset_name, exist_ok=True)

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, img_height // 2 ** 4, img_width // 2 ** 4)

# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

cuda = True if torch.cuda.is_available() else False

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(epoch, loader, mode):
    imgs = next(iter(loader))
    gray = Variable(imgs["A"].type(Tensor))
    color = Variable(imgs["B"].type(Tensor))
    output = generator(gray)

    gray_img = torchvision.utils.make_grid(gray.data, nrow=4)
    color_img = torchvision.utils.make_grid(color.data, nrow=4)
    output_img = torchvision.utils.make_grid(output.data, nrow=4)

    rows = 3
    cols = 1

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(reNormalize(gray_img.cpu(), gray_mean, gray_std))
    ax1.set_title('gray')

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(reNormalize(color_img.cpu(), color_mean, color_std))
    ax2.set_title('color')

    ax3 = fig.add_subplot(rows, cols, 3)
    ax3.imshow(reNormalize(output_img.cpu(), color_mean, color_std))
    ax3.set_title('output')

    #plt.show()
    fig.savefig("images/%s/%s/epoch_%s.png" % (dataset_name, mode, epoch), pad_inches=0)
    plt.close(fig)    

# ----------
#  Training
# ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for epoch in range(1, n_epochs+1):
    start_time = datetime.datetime.now()
    for i, batch in enumerate(train_loader):

        # Model inputs
        gray = Variable(batch["A"].type(Tensor))
        color = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(torch.tensor(np.ones((gray.size(0), *patch)), dtype=torch.float32, device=device), requires_grad=False)
        fake = Variable(torch.tensor(np.zeros((gray.size(0), *patch)), dtype=torch.float32, device=device), requires_grad=False)
        # valid = Variable(Tensor(np.ones((gray.size(0), *patch))), requires_grad=False)
        # fake = Variable(Tensor(np.zeros((gray.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        output = generator(gray)
        pred_fake = discriminator(output, gray)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(output, color)

        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(color, gray)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(output.detach(), gray)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        epoch_time = datetime.datetime.now() - start_time

    if (epoch) % checkpoint_interval == 0:
        fig = plt.figure(figsize=(18, 18))
        sample_images(epoch, train_loader, 'val')

        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (dataset_name, epoch))

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s" % (epoch,
                                                                                                    n_epochs,
                                                                                                    i+1,
                                                                                                    len(train_loader),
                                                                                                    loss_D.item(),
                                                                                                    loss_G.item(),
                                                                                                    loss_pixel.item(),
                                                                                                    loss_GAN.item(),
                                                                                                    epoch_time))

test_root = root + 'test/'
test_batch_size = 1


test_dataset = VictorianDataset(test_root)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


fig = plt.figure(figsize=(10, 5))
rows = 1
cols = 2

for X in test_loader:
    print(X['A'].shape, X['B'].shape)

    # Extract only the first channel of the gray image
    gray_img_first_channel = reNormalize(X["A"][0], gray_mean, gray_std)[:, :, 0]  # Extract 1st channel

    # Plot gray image
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(gray_img_first_channel, cmap='gray')  # Use cmap='gray' for grayscale visualization
    ax1.set_title('gray img (1st channel) after train')

    # Plot color image
    color_img = reNormalize(X["B"][0], color_mean, color_std)
    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(color_img)
    ax2.set_title('color img after train')

    #plt.show()
    break

generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (dataset_name, n_epochs)))
discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (dataset_name, n_epochs)))

def normalize_image(img):
    """Normalizes the image to the range [0, 1]."""
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

#AI 예측할 때 쓰는 sample_images, 훈련 직후 생성한 test_loader가 loader에 들어감, mode에는 'test'가 들어감
def sample_images(epoch, loader, mode):
    rows,cols = 2,3
    
    wid, hei = 924,1848
    
    imgs = next(iter(loader))
    gray = Variable(imgs["A"].type(Tensor)) #훈련 직후 생성한 test_loader부분 참고, gray 파일 불러와서 텐서로 변환
    color = Variable(imgs["B"].type(Tensor)) #[C, H, W]
    output = generator(gray) #[B, C, H, W] #[2,3,256,256]
    print(f"gray  - 최소값: {gray.min()}, 최대값: {gray.max()}")
    print(f"color  최소값: {color.min()}, 최대값: {color.max()}")
    print(f"output  최소값: {output.min()}, 최대값: {output.max()}\n")
     # 첫 번째 이미지
    gray_img_1 = gray.data.cpu().numpy()[0, 0, :, :]  # 첫 번째 배치, 첫 번째 채널(그레이스케일)
    color_img_1 = color.data.cpu().numpy()[0, :, :, :]  # 첫 번째 배치, 컬러 이미지
    output_img_1 = output.data.cpu().numpy()[0, :, :, :]  # 첫 번째 배치의 출력 이미지, 컬러임
    #정규화해서 이미지 복원할 준비
    normalized_gray1 = (gray_img_1 - gray_img_1.min()) / (gray_img_1.max()-gray_img_1.min())
    normalized_color1 = (color_img_1 - color_img_1.min()) / (color_img_1.max()-color_img_1.min())
    normalized_output1 = (output_img_1 - output_img_1.min()) / (output_img_1.max()-output_img_1.min())
    # 두 번째 이미지
    print('normalized gray min max', normalized_gray1.min(), normalized_gray1.max())
    print('normalized color min max', normalized_color1.min(), normalized_color1.max())
    print('normalized output min max', normalized_output1.min(), normalized_output1.max())
    gray_img_2 = gray.data.cpu().numpy()[1, 0, :, :]  # 두 번째 배치, 첫 번째 채널(그레이스케일)
    color_img_2 = color.data.cpu().numpy()[1, :, :, :]  # 두 번째 배치, 컬러 이미지
    output_img_2 = output.data.cpu().numpy()[1, :, :, :]  # 두 번째 배치의 출력 이미지, 컬러임
    # 두 번째 이미지도 정규화해서 할 준비

    normalized_gray2 = (gray_img_2 - gray_img_2.min()) / (gray_img_2.max()-gray_img_2.min())
    normalized_color2 = (color_img_2 - color_img_2.min()) / (color_img_2.max()-color_img_2.min())
    normalized_output2 = (output_img_2 - output_img_2.min()) / (output_img_2.max()-output_img_2.min())
    # 리사이즈
    
    resized_gray_img_1 = cv2.resize(normalized_gray1, (wid, hei), interpolation=cv2.INTER_CUBIC)  # [256, 256] -> [1848, 924]
    resized_color_img_1 = cv2.resize(normalized_color1.transpose(1,2,0), (wid, hei), interpolation=cv2.INTER_CUBIC)  # [256, 256, 3] -> [1848, 924, 3]
    resized_output_img_1 = cv2.resize(normalized_output1.transpose(1,2,0), (wid, hei), interpolation=cv2.INTER_CUBIC)  # [256, 256, 3] -> [1848, 924, 3]
    resized_gray_img_1 = np.clip(resized_gray_img_1, 0, 1) 
    resized_color_img_1 = np.clip(resized_color_img_1, 0, 1) 
    resized_output_img_1 = np.clip(resized_output_img_1, 0, 1) 
    print('resized gray min max', resized_gray_img_1.min(), resized_gray_img_1.max())
    print('resized color min max', resized_color_img_1.min(), resized_color_img_1.max())
    print('resized output min max', resized_output_img_1.min(), resized_output_img_1.max())
    resized_gray_img_2 = cv2.resize(normalized_gray2, (wid, hei), interpolation=cv2.INTER_CUBIC)  # [256, 256] -> [1848, 924]
    resized_color_img_2 = cv2.resize(normalized_color2.transpose(1,2,0), (wid, hei), interpolation=cv2.INTER_CUBIC)  # [256, 256, 3] -> [1848, 924, 3]
    resized_output_img_2 = cv2.resize(normalized_output2.transpose(1,2,0), (wid, hei), interpolation=cv2.INTER_CUBIC)  # [256, 256, 3] -> [1848, 924, 3] 
    resized_gray_img_2 = np.clip(resized_gray_img_2, 0, 1) 
    resized_color_img_2 = np.clip(resized_color_img_2, 0, 1) 
    resized_output_img_2 = np.clip(resized_output_img_2, 0, 1) 
    
    # 첫 번째 배치 (첫 번째 이미지들)
    plt.subplot(rows, cols, 1)
    plt.imshow(resized_gray_img_1, cmap='gray')  # 그레이스케일 이미지는 'gray' 컬러맵
    plt.title('Gray 1')

    plt.subplot(rows, cols, 2)
    plt.imshow(resized_color_img_1)  # 컬러 이미지는 'RGB'
    plt.title('Color 1')
    
    plt.subplot(rows, cols, 3)
    plt.imshow(resized_output_img_1)  # 생성된 이미지
    plt.title('Output 1')

    # 두 번째 배치 (두 번째 이미지들)
    plt.subplot(rows, cols, 4)
    plt.imshow(resized_gray_img_2, cmap='gray')  # 그레이스케일 이미지
    plt.title('Gray 2')
    
    plt.subplot(rows, cols, 5)
    plt.imshow(resized_color_img_2)  # 컬러 이미지
    plt.title('Color 2')

    plt.subplot(rows, cols, 6)
    plt.imshow(resized_output_img_2)  # 생성된 이미지
    plt.title('Output 2')

    #plt.show()
    fig.savefig("images/%s/%s/epoch_%s.png" % (dataset_name, mode, epoch), pad_inches=0)
    
    
    # plt.close(fig)
    # print(f"gray img - 최소값: {gray_img_1.min()}, 최대값: {gray_img_1.max()}")
    # print(f"color img 최소값: {color_img_1.min()}, 최대값: {color_img_1.max()}\n")
    # print('gray img 1 shape', gray_img_1.shape)
    # print('color img 1 shape', color_img_1.shape)
    # print('output img 1 shape', output_img_1.shape) 

generator.eval()
discriminator.eval()

fig = plt.figure(figsize=(15,10)) #최종 리사이즈 복원할 때 여기 수정+저 위에 reNormalize 수정파트랑 같이(?)
sample_images(n_epochs, test_loader, 'test')

imgs = next(iter(test_loader))
gray = Variable(imgs["A"].type(Tensor))
color = Variable(imgs["B"].type(Tensor))
output = generator(gray)

gray=gray[0]
color=color[0]
output=output[0]
print('output shape', output.shape)
output_mean = output.mean(dim=[1, 2])  # 각 채널에 대한 평균값, shape: [3]
output_std = output.std(dim=[1, 2])    # 각 채널에 대한 표준편차, shape: [3]
output_mean_list = output_mean.tolist()
output_std_list = output_std.tolist()
print('Output mean:', output_mean_list)
print('Output std:', output_std_list)
print('color mean:', color_mean)
print('color std:', color_std)
a=reNormalize(output.detach().cpu(),output_mean_list,output_std_list)
print('gray renormalized shape', a.shape)
b=reNormalize(color.detach().cpu(),color_mean,color_std)
mse=(np.square(a-b)).mean(axis=None)
print(mse)

# # gray와 color 이미지의 Min-Max 값을 계산
# output_min = output.min().item()  
# output_max = output.max().item()  

# color_min = color.min().item() 
# color_max = color.max().item()  
# print("output Min:", output_min, "output Max:", output_max)
# print("Color Min:", color_min, "Color Max:", color_max)
# # min max 복원 함수
# def reNormalize_minmax(normalized_tensor, min_value, max_value):
#     return normalized_tensor * (max_value - min_value) + min_value

# # Min-Max 복원
# a = reNormalize_minmax(output.detach().cpu(), output_min, output_max)
# b = reNormalize_minmax(color.detach().cpu(), color_min, color_max)

# # MSE 계산
# mse = (np.square(a - b)).mean(axis=None)
# print(mse.item())