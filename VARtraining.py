import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
num_classes = 10 # datasets have image file of number 0 to 9
epochs = 15
root_path = "C:\\Users\\김해창\\Desktop\\VAR\data"

#리사이징, 텐서로 변환 
transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#train=True: 훈련 데이터셋/train=False: 테스트 데이터셋/transform=transform: 위에거 적용/download=True: 로컬에 데이터셋 없으면 자동으로 다운로드드
#내가 만든게 아니고 제공되는 데이터셋을 다운받아오는 거
dataset = datasets.MNIST(root=root_path, train=True, transform=transform, download=True)

train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

#데이터셋 배치 단위로 나눠서 모델에 공급,shuffle: 섞을지 말지
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

for images, labels in train_loader: #위에서는 데이터
    print(f"Shape of images: {images.shape}") # (128, 1, 28, 28) (batchsize, channels, height, width)
    print(f"Shape of labels: {labels.shape}") # 128
    break

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()  #nn.Module initialization 작업: 상속하는듯?
        self.fc1 = nn.Linear(784, 512) # (input vector(28x28), output) fully connected layer하나
        self.fc2 = nn.Linear(512, 512) # output개수랑 뉴런개수는 동일
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.2) #output개수 = 뉴런 개수 512개 각각 0.2의 확률로 비활성화됨

    def forward(self, x):   #레이어 하나하나 전진하는 부분분
        x = x.view(-1, 784) # (batch size, 28, 28) -> (batch size, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 윗 줄에서 만든 512개의 값 중 약 20% 정도가 0의 값을 가지는 세트로 조정됨됨
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(784, 512) # (input vector, output)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = x.view(-1, 784) # (batch size, 28, 28) -> (batch size, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
model1 = Net1()
model2 = Net2()
#Net1,2는 파라미터 조정할 거라서 상속받아와서 정의, 근데 손실함수랑 옵티마이저는 계산만 수행하는(?) 느낌이라 그냥 불러와서 씀
#파라미터 손댈 일 없어서 def로 굳이 안함
def training(model, train_loader,epochs,criterion,optimizer):    
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad() #파이토치는 기울기가 누적되서 여기서 리셋함
            loss.backward() #위에서 criterion을 이용해서 계산한 loss값으로 기울기를 계산, model.parameter()에 저장됨
            optimizer.step() #위에서 계산한 기울기로 파라미터 업데이트, optimizer 모델은 여러개있음 이 코드에서는 optim.RMSprop
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{str(epoch+1).zfill(2)}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        # save model
    torch.save(model.state_dict(), f'C:\\Users\\김해창\\Desktop\\VAR\\{model.__class__.__name__}.pth')

if __name__ == '__main__':
    model1 = Net1()
    criterion = nn. CrossEntropyLoss() #예측값과 실제값의 차이를 계산하는 내용의 코드가 들어있는 함수()
    optimizer = optim.RMSprop(model1.parameters())
    training(model1,train_loader,epochs,criterion,optimizer)
    
    model2 = Net2()
    optimizer = optim.RMSprop(model2.parameters())
    training(model2,train_loader,epochs,criterion,optimizer)