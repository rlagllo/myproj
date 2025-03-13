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
from var_resnet_paper import ResidualBlock, IdentityPadding, resnet, ResNet, mydataset

testx_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\test_x.npz'
testy_path = r'C:\Users\김해창\Desktop\cifar-10-batches-py\refined\test_y.npz'

transforms_test = transforms.Compose([  # 일반적으로 이미지는 (H,W,C)의 형식인데(numpy&PIL) 텐서로 바꿈(C,H,W)
    #transforms.Grayscale(),
    #transforms.Resize((128, 128), interpolation= transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #transforms.Lambda(lambda img: img.convert("RGB"))
])


testdataset = mydataset(testx_path, testy_path, transform=transforms_test)
test_loader = DataLoader(testdataset, batch_size=8, shuffle=True)
model_path = "C:\\Users\\김해창\Desktop\\cifar-10-batches-py\\model_resnet.pth"


net = resnet()
net = net.to('cuda')

state_dict = torch.load(model_path, weights_only=True)

net.load_state_dict(state_dict, strict=True)

net.eval()
summary(net, input_size= (3,32,32))

correct = 0 # 맞춘 문제 수
total = 0 # 총 문제 수
answer_match = []
classes =  ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        classification, feature = net(images)
        _, prediction = torch.max(classification,1)
        print(images.shape)
        print(labels.shape)
        print(prediction.shape)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()
        params = list(net.parameters())[-2]
        overlay = params[int(prediction[1])].matmul(feature[1].reshape(64,8*8)).reshape(8,8).cpu().data.numpy() # 히트맵 8 * 8, 64 & 64, 10 -> 8*8 , 10
        overlay = overlay - np.min(overlay)
        overlay = overlay / np.max(overlay) # heatmap 만들기, 정규화과정
        overlay_resized = skimage.transform.resize(overlay, [128, 128]) #원본 이미지와 같은 크기로 만들어서 겹쳐서 시각화하려고
        original_image = images[1].cpu().numpy()
        img = np.transpose(original_image, (1, 2, 0)) # permute와 같은 효과
        img = skimage.transform.resize(img,[128,128])

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        
        #img = original_image.permute(1, 2, 0) # [C,H,W]인 이미지 텐서형식을 [H,W,C]로 만들어서 matplotlib할거임
        #img = (img - np.min(img)) / (np.max(img) - np.min(img)) #정규화, 원본 이미지이지만 matplotlib은 정규화해서 시각화하는게 일반적 + 정규화시킨 heatmap과 맞추기 위해
        ax[0].imshow(img)
        ax[0].set_title(f"Original Image & answer is {classes[labels[1]]}")
        ax[0].axis('off')    
        ax[1].imshow(img)
        ax[1].imshow(overlay_resized, alpha=0.4, cmap='jet')
        ax[1].set_title(f"Learned Overlay & predicted as: {classes[int(prediction[1])]}")
        ax[1].axis('off')       
        plt.show()         
    
    accu = 100 * (correct / total)
    print(f"test accu: {accu:.2f}%")    
        








# def test(epoch, best_acc, global_steps):
#     net.eval()

#     test_loss = 0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(test_loader):
#             inputs = inputs.to('cuda')
#             targets = targets.to('cuda')
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#     acc = 100 * correct / total
#     print('test epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
#         epoch, batch_idx, len(test_loader), test_loss / (batch_idx + 1), acc))

#     writer.add_scalar('log/test error', 100 - acc, global_steps)

#     if acc > best_acc:
#         print('==> Saving model..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('save_model'):
#             os.mkdir('save_model')
#         torch.save(state, './save_model/ckpt.pth')
#         best_acc = acc

#     return best_acc


        
        