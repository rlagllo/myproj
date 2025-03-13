from z_UNetprac import unet
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from z_UNetprac import NiftiDataset 
from PIL import Image
import matplotlib.pyplot as plt


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


test_dataset = NiftiDataset(image_dir=test_dir, label_dir=label_dir, image_transform=transform, label_transform=transform2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = unet(input_channels=4,output_channels=4).to(device)

model.load_state_dict(torch.load("C:\\Users\\김해창\\Desktop\\Task01_BrainTumour\\model.pth"))

model.eval()

for inputs,_ in test_loader:
    
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
    input_image = inputs[0].cpu().numpy()    
    predictions = torch.argmax(outputs,dim=1) # 각 픽셀에 대해 레이블값 (0,1,2,3)중에 뭔지 할당
    print(predictions)
    predict_mask = predictions[0].cpu().numpy()
    input_image_vis = input_image[0]  # 첫 번째 채널 선택 (H, W)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(input_image_vis, cmap="gray")  # 원본 이미지 (흑백)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(predict_mask, cmap="jet")  # 0,1,2,3에 대해 색상 할당
    plt.title("Predicted Segmentation Mask")
    plt.axis("off")

    plt.show()