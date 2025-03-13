import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from keras.datasets import cifar10
from var_cnn_network import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 30
learning_rate = 0.0003
num_epoch = 10


# CIFAR-10 이미지는 (50000(10000) 32, 32, 3) (이미지 개수 너비 높이 RGB채널), 0-255 범위의 픽셀값
# 32 32 3중에서 컬러 3채널만 0~1사이의 값으로 변형되고, 3 32 32로 shape바뀜
# 채널이 어디에 위치하든 ToTensor()는 그 위치를 자동으로 감지하여 변환
# # CIFAR-10 평균, 표준편차: 대강 알려진 값임
 
transform = transforms.Compose([
    transforms.ToTensor(),  # ToTensor()가 먼저, 그 후 Normalize
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomHorizontalFlip(),  # 좌우 반전
    transforms.RandomRotation(15),  # 15도 회전
])
# x는 2D데이터, y는 레이블(정답)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()  #50000(10000) 3 32 32 shape임 원래 비중을 두고 테스트용이랑 훈련용 나눠야 하는데 자동으로 해줌
x_train_transformed = torch.stack([transform(image) for image in x_train]) #4D 배열의 각각 3D 배열에 대해 transform 진행 32 32 3 -> 3 32 32가 50000번 일어남
x_test_transformed = torch.stack([transform(image) for image in x_test])
print(x_train_transformed.shape)
y_train = torch.tensor(y_train.squeeze(), dtype=torch.long)
y_test = torch.tensor(y_test.squeeze(), dtype=torch.long)
print(x_train_transformed.shape)
print(y_train.shape)

train_dataset = TensorDataset(x_train_transformed, y_train)
test_dataset = TensorDataset(x_test_transformed, y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = CNN(batch_size=batch_size).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("트레이닝 시작")
loss_arr =[]
for i in range(num_epoch):
    epoch_loss = 0
    corrects = 0
    total = 0
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y= label.to(device)
        
        optimizer.zero_grad()
        
        output = model.forward(x)
        
        loss = loss_func(output,y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(output, 1)
        corrects += (predicted == y).sum().item()
        total += y.size(0)

    epoch_loss /= len(train_loader)
    epoch_acc = corrects / total * 100
    print(f"epoch: [{str(i+1).zfill(2)}/{num_epoch}], Loss: {epoch_loss:.5f}, Accuracy: {epoch_acc:.5f}")

correct = 0
total = 0
model.eval()
#채점
with torch.no_grad(): 
    for image,label in test_loader:
        x, y = image.to(device), label.to(device)
        output = model.forward(x)

        _,output_index = torch.max(output,1)

        total += label.size(0)
        correct += (output_index == y).sum().float()

    print(f"Accuracy of Test Data: {100*correct/total:.5f}%")
    
# 이것 또한 다른 곳에서 얻어온 데이터셋을 가지고, 다