import os
os.chdir("C:\\Users\\김해창\\Desktop\\project")

import glob
import numpy as np
import datetime
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity as ssim  # Ensure this import is included
from torchvision.models import vgg16

root = ''

batch_size = 4
img_height = 256
img_width = 256

def calculate_min_max(image_paths):
    min_vals = torch.tensor([float('inf')] * 3)
    max_vals = torch.tensor([float('-inf')] * 3)

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensor = ToTensor()(img)
        min_vals = torch.minimum(min_vals, tensor.view(3, -1).min(dim=1)[0])
        max_vals = torch.maximum(max_vals, tensor.view(3, -1).max(dim=1)[0])

    return min_vals.tolist(), max_vals.tolist()

# Calculate for gray, color, and grey datasets
gray_paths = sorted(glob.glob(os.path.join(root, 'gray') + "/*.*"))
color_paths = sorted(glob.glob(os.path.join(root, 'resized') + "/*.*"))
grey_paths = sorted(glob.glob(os.path.join(root, 'grey') + "/*.*"))

gray_min, gray_max = calculate_min_max(gray_paths)
color_min, color_max = calculate_min_max(color_paths)
grey_min, grey_max = calculate_min_max(grey_paths)

print("Gray Min:", gray_min, "Gray Max:", gray_max)
print("Color Min:", color_min, "Color Max:", color_max)
print("Grey Min:", grey_min, "Grey Max:", grey_max)

class VictorianDataset(Dataset):
    def __init__(self, root, img_size=(256, 256), gray_min=None, gray_max=None, color_min=None, color_max=None, grey_min=None, grey_max=None):
        self.img_size = img_size
        self.gray_min = torch.tensor(gray_min).view(3, 1, 1)
        self.gray_max = torch.tensor(gray_max).view(3, 1, 1)
        self.color_min = torch.tensor(color_min).view(3, 1, 1)
        self.color_max = torch.tensor(color_max).view(3, 1, 1)
        self.grey_min = torch.tensor(grey_min).view(3, 1, 1)
        self.grey_max = torch.tensor(grey_max).view(3, 1, 1)

        self.gray_files = sorted(glob.glob(os.path.join(root, 'gray') + "/*.*"))
        self.color_files = sorted(glob.glob(os.path.join(root, 'resized') + "/*.*"))
        self.grey_files = sorted(glob.glob(os.path.join(root, 'grey') + "/*.*"))

    def __getitem__(self, index):
        gray_img = Image.open(self.gray_files[index % len(self.gray_files)]).convert("RGB")
        color_img = Image.open(self.color_files[index % len(self.color_files)]).convert("RGB")
        grey_img = Image.open(self.grey_files[index % len(self.grey_files)]).convert("RGB")

        gray_img = gray_img.resize(self.img_size, Image.BICUBIC)
        color_img = color_img.resize(self.img_size, Image.BICUBIC)
        grey_img = grey_img.resize(self.img_size, Image.BICUBIC)

        gray_tensor = transforms.ToTensor()(gray_img)
        color_tensor = transforms.ToTensor()(color_img)
        grey_tensor = transforms.ToTensor()(grey_img)

        gray_tensor = (gray_tensor - self.gray_min) / (self.gray_max - self.gray_min)
        color_tensor = (color_tensor - self.color_min) / (self.color_max - self.color_min)
        grey_tensor = (grey_tensor - self.grey_min) / (self.grey_max - self.grey_min)

        return {"A": gray_tensor, "B": color_tensor, "C": grey_tensor}

    def __len__(self):
        return len(self.gray_files)

train_loader = DataLoader(
    VictorianDataset(root, img_size=(256, 256), gray_min=gray_min, gray_max=gray_max, color_min=color_min, color_max=color_max, grey_min=grey_min, grey_max=grey_max),
    batch_size=batch_size,
    shuffle=True
)

def reNormalize(img, min_vals, max_vals):
    # Convert input min_vals and max_vals to numpy arrays for broadcasting
    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)

    img = img.numpy().transpose(1, 2, 0)  # Convert to HWC format
    img = img * (max_vals - min_vals) + min_vals  # De-normalize
    img = img.clip(0, 1)  # Ensure values are within [0, 1]
    return img

# Ensure min/max values are tensors or arrays
gray_min, gray_max = torch.tensor(gray_min), torch.tensor(gray_max)
color_min, color_max = torch.tensor(color_min), torch.tensor(color_max)

fig = plt.figure(figsize=(10, 5))
rows, cols = 1, 2

for X in train_loader:
    # De-normalize images using reNormalize
    gray_img = reNormalize(X["A"][0], gray_min, gray_max)
    color_img = reNormalize(X["B"][0], color_min, color_max)

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(gray_img)
    ax1.set_title('Gray Image')

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(color_img)
    ax2.set_title('Color Image')

    plt.show()
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

# Modify GeneratorUNet to accept additional input channel
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):  # Increase in_channels to 6 (gray + grey)
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

n_epochs = 500
dataset_name = "Victorian400"
lr = 0.0002
b1 = 0.5                    # adam: decay of first order momentum of gradient
b2 = 0.999                  # adam: decay of first order momentum of gradient
decay_epoch = 100           # epoch from which to start lr decay
n_cpu = 8                   # number of cpu threads to use during batch generation
channels = 3                # number of image channels
checkpoint_interval = 50    # interval between model checkpoints

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

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features)[:16])  # Use up to relu2_2
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = self.features(x)
        y_features = self.features(y)
        return F.mse_loss(x_features, y_features)

# Initialize perceptual loss
criterion_perceptual = PerceptualLoss()
if cuda:
    criterion_perceptual = criterion_perceptual.cuda()

def sample_images(epoch, loader, mode):
    imgs = next(iter(loader))
    gray = Variable(imgs["A"].type(Tensor))
    color = Variable(imgs["B"].type(Tensor))
    grey = Variable(imgs["C"].type(Tensor))

    combined_input = torch.cat((gray, grey), 1)  # Combine gray and grey inputs
    output = generator(combined_input)

    gray_img = torchvision.utils.make_grid(gray.data, nrow=6)
    grey_img = torchvision.utils.make_grid(grey.data, nrow=6)
    color_img = torchvision.utils.make_grid(color.data, nrow=6)
    output_img = torchvision.utils.make_grid(output.data, nrow=6)

    rows = 4
    cols = 1

    fig = plt.figure(figsize=(18, 18))
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(reNormalize(gray_img.cpu(), gray_min, gray_max))
    ax1.set_title('gray')

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(reNormalize(grey_img.cpu(), grey_min, grey_max))
    ax2.set_title('grey')

    ax3 = fig.add_subplot(rows, cols, 3)
    ax3.imshow(reNormalize(color_img.cpu(), color_min, color_max))
    ax3.set_title('color')

    ax4 = fig.add_subplot(rows, cols, 4)
    ax4.imshow(reNormalize(output_img.cpu(), color_min, color_max))
    ax4.set_title('output')

    plt.show()
    fig.savefig("images/%s/%s/epoch_%s.png" % (dataset_name, mode, epoch), pad_inches=0)

test_root = root + 'test/'
test_batch_size = 4


test_loader = DataLoader(
    VictorianDataset(
        test_root,
        img_size=(256, 256),
        gray_min=gray_min,
        gray_max=gray_max,
        color_min=color_min,
        color_max=color_max,
        grey_min=grey_min,
        grey_max=grey_max  # 추가
    ),
    batch_size=test_batch_size,
    shuffle=False
)

def evaluate_model(generator, loader, gray_min, gray_max, color_min, color_max):
    generator.eval()

    mse_values = []
    ssim_values = []

    for batch in loader:
        # Extract gray and target color images
        gray = batch['A'].type(Tensor)
        grey = batch['C'].type(Tensor)
        color_target = batch['B'].type(Tensor)

        # Combine inputs and generate output
        combined_input = torch.cat((gray, grey), 1)
        with torch.no_grad():
            color_output = generator(combined_input)

        # De-normalize target and output images for evaluation
        color_target_denorm = reNormalize(color_target[0].cpu(), color_min, color_max)
        color_output_denorm = reNormalize(color_output[0].cpu(), color_min, color_max)

        # Compute MSE
        mse = F.mse_loss(torch.tensor(color_target_denorm), torch.tensor(color_output_denorm))
        mse_values.append(mse.item())

        # Compute SSIM
        ssim_score = ssim(
            color_target_denorm,
            color_output_denorm,
            multichannel=True,
            data_range=1.0,
            win_size=3,  # Adjust the win_size to fit image size
            channel_axis=-1  # Specify channel axis for color images
        )
        ssim_values.append(ssim_score)

    # Average MSE and SSIM
    avg_mse = sum(mse_values) / len(mse_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)

    print("Average MSE:", avg_mse)
    print("Average SSIM:", avg_ssim)

# Evaluate on test data
evaluate_model(generator, test_loader, gray_min, gray_max, color_min, color_max)

# ----------
#  Training
# ----------
# Update training loop
for epoch in range(1, n_epochs + 1):
    start_time = datetime.datetime.now()
    for i, batch in enumerate(train_loader):
        gray = Variable(batch["A"].type(Tensor))
        color = Variable(batch["B"].type(Tensor))
        grey = Variable(batch["C"].type(Tensor))

        combined_input = torch.cat((gray, grey), 1)  # Combine gray and grey inputs

        valid = Variable(Tensor(np.ones((gray.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((gray.size(0), *patch))), requires_grad=False)

        # Train Generators
        optimizer_G.zero_grad()
        output = generator(combined_input)
        pred_fake = discriminator(output, gray)
        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_pixel = criterion_pixelwise(output, color)
        loss_perceptual = criterion_perceptual(output, color)  # Perceptual loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel + 0.01 * loss_perceptual  # Weighted perceptual loss
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        pred_real = discriminator(color, gray)
        loss_real = criterion_GAN(pred_real, valid)
        pred_fake = discriminator(output.detach(), gray)
        loss_fake = criterion_GAN(pred_fake, fake)
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        epoch_time = datetime.datetime.now() - start_time

    if (epoch) % checkpoint_interval == 0:
        fig = plt.figure(figsize=(18, 18))
        sample_images(epoch, train_loader, 'val')

        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (dataset_name, epoch))

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f, perceptual: %f] ETA: %s" % (
            epoch,
            n_epochs,
            i+1,
            len(train_loader),
            loss_D.item(),
            loss_G.item(),
            loss_pixel.item(),
            loss_GAN.item(),
            loss_perceptual.item(),
            epoch_time))

fig = plt.figure(figsize=(10, 5))
rows = 1
cols = 2

for X in test_loader:

    print(X['A'].shape, X['B'].shape)
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(reNormalize(X["A"][0], gray_min, gray_max))
    ax1.set_title('gray img')

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(reNormalize(X["B"][0], color_min, color_max))
    ax2.set_title('color img')

    plt.show()
    break

generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (dataset_name, n_epochs)))
discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (dataset_name, n_epochs)))

generator.eval()
discriminator.eval()

fig = plt.figure(figsize=(18,10))
sample_images(n_epochs, test_loader, 'test')

# Evaluate on test data
evaluate_model(generator, test_loader, gray_min, gray_max, color_min, color_max)