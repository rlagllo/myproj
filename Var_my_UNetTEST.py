from Var_myproject_UNet import mydataset, unet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
from torchsummary import summary
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "C:\\Users\\김해창\Desktop\\cifar-10-batches-py\\model_gray.pth"
testx_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\test_x.npz'
testy_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\test_y.npz'

transforms = transforms.Compose([  # 일반적으로 이미지는 (H,W,C)의 형식인데(numpy&PIL) 텐서로 바꿈(C,H,W)
    transforms.Grayscale(num_output_channels= 1), # 흑백으로 바꾸기
    transforms.Resize((128, 128), interpolation= transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # 컬러일때 
    transforms.Normalize((0.5),(0.5)) # 흑백일 떄
]) 

state_dict = torch.load(model_path,weights_only=True)

model = unet(1,1,10).to(device)
model.load_state_dict(state_dict,strict=True)

test_dataset = mydataset(testx_path, testy_path, transform=transforms)
test_dataloader = DataLoader(test_dataset,batch_size=64)

model.eval()
with torch.no_grad():
    for images, _ in test_dataloader:
        images = images.to(device)

        restored = model(images)
        toshow = restored[0].permute(1,2,0).cpu().numpy()
        original = images[0].permute(1,2,0).cpu().numpy()
        
    fig,axes = plt.subplots(1,2,figsize= (12,4))
    axes[0].imshow(original)
    axes[0].set_title(f"Original Image")
    axes[0].axis("off")

    # 복원된 이미지
    axes[1].imshow(toshow)
    axes[1].set_title("Restored Image")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()
        
        
        