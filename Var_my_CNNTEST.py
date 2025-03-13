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
import glob
from VAR_myproj_CNN import mycnn, mydataset, transforms_test
from torchsummary import summary
# transforms = transforms.Compose([  # 일반적으로 이미지는 (H,W,C)의 형식인데(numpy&PIL) 텐서로 바꿈(C,H,W)
#     transforms.Resize((128, 128), interpolation= transforms.InterpolationMode.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
# ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "C:\\Users\\김해창\Desktop\\cifar-10-batches-py\\model_gray_CNN.pth"
testx_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\test_x.npz'
testy_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\test_y.npz'
# 저장된 모델 가중치 로드
state_dict = torch.load(model_path, weights_only=True)  # FutureWarning 해결 위해 weights_only=True 사용



# 현재 모델 선언
model = mycnn(1, 1, 10).to(device)
model.load_state_dict(state_dict,strict=True)
model.eval()
summary(model, input_size= (1,128,128))



# 현재 모델의 키 출력
#print("현재 모델 키:", model.state_dict().keys())

# 저장된 모델의 키 출력
#print("저장된 모델 키:", state_dict.keys())
#model.eval()
test_dataset = mydataset(testx_path, testy_path,transforms_test)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
classes =  ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
correct = 0 # 맞춘 문제 수
total = 0 # 총 문제 수
answer_match = []
with torch.no_grad():
    it = 0
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)           
            # classification은 한 배치(64개)에 대해 64개 사진 중 1번째: [0.02,0.1,0.5,,,,,,] 총 10개 인덱스의 softmax값을 가짐 -> [[0.02,0.1,0.5,,,,], [0.04,0.2,0.3,0.01,,,,], [----]] 형태
            # 그래서 torch.max(classification)은 batch_size, labels니까, 최대인 애의 확률값이 아니고 정답값을 가리키는 차원을 써야함 배치단위일 때 dim = 0을 하면 최대인 애의 확률값을 가져오고, dim=1이면 최대인 애의 인덱스(레이블)값을 가져옴
                   
        classification,features = model(images)
        _, prediction = torch.max(classification, dim=1)
           
        #print("predicted: ", prediction)
        #print("labels: ", labels)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()
        #params = list(model.parameters())[-2]
      
    #overlay = params[int(prediction[1])].matmul(features[1].reshape(512,16*16)).reshape(16,16).cpu().data.numpy() # 히트맵 10,512  matmul   
    #Aoverlay = overlay - np.min(overlay)
    #overlay = overlay / np.max(overlay) # heatmap 만들기, 정규화과정
    #overlay_resized = skimage.transform.resize(overlay, [128, 128]) #원본 이미지와 같은 크기로 만들어서 겹쳐서 시각화하려고

    original_image = images[1].cpu()    

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    img = original_image.permute(1, 2, 0).numpy() # [C,H,W]인 이미지 텐서형식을 [H,W,C]로 만들어서 matplotlib할거임
    #img = (img - np.min(img)) / (np.max(img) - np.min(img)) #정규화, 원본 이미지이지만 matplotlib은 정규화해서 시각화하는게 일반적 + 정규화시킨 heatmap과 맞추기 위해
    ax[0].imshow(img,cmap = 'gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')    
    ax[1].text(0.5, 0.5, f"Predicted Class: {classes[prediction[1].item()]}", 
    fontsize=20, ha="center", va="center")
    #ax[1].imshow(overlay_resized, alpha = 0.4, cmap='jet')
    ax[1].set_title("Learned Overlay")
    ax[1].axis('off')       
    plt.show()    
        
        
        # if (it < 15):
        #     fig,axes = plt.subplots(1,2,figsize= (12,4))
        #     axes[0].imshow(images[0].cpu().permute(1,2,0).numpy())
        #     axes[0].set_title(f"Original Image (Label: {labels[0].item()})")
        #     axes[0].axis("off")
        #     #예측된 클래스 표시
        #     axes[1].text(0.5, 0.5, f"Predicted Class: {prediction[0].item()}", 
        #                 fontsize=20, ha="center", va="center")
        #     axes[1].set_title("Classification Result")
        #     axes[1].axis("off")
            # plt.tight_layout()
            # plt.show()
            
accu = 100 * (correct / total)
print(f"test accu: {accu:.2f}%")
    
    
    
# params_list =list(model.parameters())

# # for name, module in model.named_modules():
# #     print(name)    
    
# # coco test
    
# pics = glob.glob("C:\\Users\\김해창\\Desktop\\cifar-10-batches-py\\coco\\강코코*")
# processed = []
# for img in pics:
#     imgg = Image.open(img).convert("RGB")
#     processed.append(transforms1(imgg))
# processed = torch.stack(processed)
    
# pics1 = Image.open(pics[0]).convert("RGB")
# processed_images = transforms1(pics1)
# #print(pics)
# #print(processed_images.shape)
# classes =  ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
# for pic in processed:
#     #pic = torch.rot90(pic, k=-1, dims=(1,2))
#     # img = pic.permute(1,2,0).cpu().numpy()
#     # plt.imshow(img)
#     # plt.show()
#     pic = pic.unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         classification, features = model(pic)
#         predicted = torch.argmax(classification,dim = 1) # 0~~9
#         print(classes[predicted.item()])
        
#     params = list(model.parameters())[-2] #  torch.Size([10, 512]) 
#     print("\nparams_list: ", params_list[-2],"          \n전체길이:   ",len(list(model.parameters())))    # 파라미터의 가장 마지막 2개는 일반적으로 weight, bias임 여기서 필요한 건 weight니까 -2번째 인덱스
#     print("\nfeatures.shape", features[0].shape,'\n')
#     print("\nparams.shape: ",params.shape,"\n")
#     #print("list(model.parameters()): ",list(model.parameters()))
#     for p in pic:
#         for predict in predicted:
#             for feature in features:
#                 overlay = params[int(predict)].matmul(feature.reshape(512,16*16)).reshape(16,16).cpu().data.numpy() # 히트맵 10,512  matmul   
#                 Aoverlay = overlay - np.min(overlay)
#                 overlay = overlay / np.max(overlay) # heatmap 만들기, 정규화과정
#                 overlay_resized = skimage.transform.resize(overlay, [128, 128]) #원본 이미지와 같은 크기로 만들어서 겹쳐서 시각화하려고

#                 original_image = pic[0].cpu()    

#                 fig, ax = plt.subplots(1, 2, figsize=(10, 5))

#                 img = original_image.permute(1, 2, 0).numpy() # [C,H,W]인 이미지 텐서형식을 [H,W,C]로 만들어서 matplotlib할거임
#                 img = (img - np.min(img)) / (np.max(img) - np.min(img)) #정규화, 원본 이미지이지만 matplotlib은 정규화해서 시각화하는게 일반적 + 정규화시킨 heatmap과 맞추기 위해
#                 ax[0].imshow(img)
#                 ax[0].set_title("Original Image")
#                 ax[0].axis('off')    
#                 ax[1].imshow(img)
#                 ax[1].imshow(overlay_resized, alpha=0.4, cmap='jet')
#                 ax[1].set_title("Learned Overlay")
#                 ax[1].axis('off')       
#                 plt.show()
