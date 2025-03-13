import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CIFAR10, ImageNet, STL10
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import skimage.transform

num_epochs = 15

transform = transforms.Compose([  # 일반적으로 이미지는 (H,W,C)의 형식인데(numpy&PIL) 텐서로 바꿈(C,H,W)
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = STL10(root='.venv\data', split='train', download=True, transform=transform)
test_dataset = STL10(root='.venv\data', split='test', download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Dataset size: {len(train_loader)}")
image, label = next(iter(train_loader))
print(f"Image shape: {image.shape}")

# VGG16은 CNN을 개조하여, 13개의 conv layer(features부분), 3개의 fc layer(classifier부분)을 만들어놓은 애인데,
# base_model.classifier = nn.Identity()를 통해 3개의 fc layer(classifier부분)을 거세하고, 
# 아래 self.base = base_model.features를 통해 13개의 conv layer(features부분)만 취하고, 
# self.fc = nn.Linear(512, 10)를 추가해서 거세된 classifier부분에 직접 정의한 fc layer를 넣고,
# self.gap = nn.AvgPool2d(7)를 추가해서, CNN을 개조한 VGG16을 개조한 VGG16_with_GAP(nn.Module)를 만들어 내는 것것
base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
base_model.classifier = nn.Identity()

class VGG16_with_GAP(nn.Module): # 원래 거 쓰면 되는데 원래 거에는 없는 GAP(풀링) 추가하려고 커스텀 클래스로 정의함
    def __init__(self):
        super(VGG16_with_GAP, self).__init__() # super로 부모 클래스(nn.Module) 호출, 커스텀 모델 만들려면 이거 해야함
        self.base = base_model.features 
        # VGG16모델의 특징추출부분만 가져옴: base_model.features. VGG16은 사전 학습된 모델임. 
        # 여기서 convolution 레이어만 가져오는 거, 원래는 마지막에 FC가 있는데 여기서는 그거 안가져온거임
        self.gap = nn.AvgPool2d(7) # GAP(global average pooling) 층 정의, nn.AvgPool2d(7): 7x7 크기의 풀링 영역을 계산, 사이즈를 줄이는 풀링.
        self.fc = nn.Linear(512, 10) # 입력 512채널->출력10채널의 FC정의, 여기가 classifier부분이라고 하면 될듯
        self.softmax = nn.Softmax(dim=1) # 출력을 확률분포로 변환 dim=1: FC의 출력이 10개니까 그 10개의 확률이 합쳐져서 1이되는 분포가 될 것임

        for param in list(self.base.parameters())[:-4]: #일반적으로 convolution layer들이 가지고 있는 가중치와 편향값들인데, 이 여러 layer들중 마지막 4개 빼고는 추후에도 학습되어 변경되지 않게 하는 것
            #self.base=base_model.features니까,
            #self.base.parameters는 base_model.features의 파라미터를 반환. [:-4]: 마지막 4개 층을 제외하고 모든 파라미터
            #위에서 선택한 파라미터의 requires_grad를 False로 설정, 역전파시 업데이트 안됨. 
            param.requires_grad = False

    def forward(self, x):
        x = self.base(x) #self.base = base_model.features의 내용 수행, 13개의 conv layer통과시킴
        features = x.clone() #추출한 피처를 저장, conv layer만 통과시킨 결과물
        x = self.gap(x) #GAP 적용( 풀링 실시)
        x = x.view(x.size(0), -1) # 1D로 펼침, 텐서의 shape를 바꿈, flatten
        #x.size(0)은 보통 배치사이즈임(batch_size, channels, height, width), 여기서는 
        x = self.fc(x) # FC 통과
        x = self.softmax(x) # softmax로 확률분포로써 다룸
        return x, features # 결과x와 피처 반환

model = VGG16_with_GAP()

for name, param in model.named_parameters(): # 파라미터 확인하기
    print(name, param.shape)


# 만약 배치크기가 10이면, model은 10개에 대해 동시에 전부 처리. criterion이 그 10개에 대해 적용, 손실값 전부 나옴
# 배치 loss는 평균값으로 계산, 이걸 기반으로 기울기 계산(배치내 10개의 loss의 평균으로 구한 기울기) backward()호출하면 기울기 계산
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss().cuda() #손실함수, outputs와 labels(답지)의 차이 계산, .backward()에서 소환됨
optimizer = optim.Adam(model.parameters(),lr=0.00001) # 기울기 기반으로 가중치(파라미터) 업데이트

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        #print(inputs.device, labels.device)

        optimizer.zero_grad() # 기울기 초기화.

        outputs, _ = model(inputs) # 여기서는 __call__메서드 호출. __call__은 정의된 forward를 호출함. 이 한 줄에서 초기화와 모든 계산이 끝남
        loss = criterion(outputs, labels) # 위에서 정의한 대로nn.CrossEntropyLoss를 이용해 손실 값 반환. 값 1개만 나옴. 애초에 outputs가 forward 정의에서 softmax 적용돼서 나오는데, 권장되지 않는다고 함. 알아봐야됨
        loss.backward() #기울기 계산. "model.parameter" 에 저장됨. 텐서 형태로, grad속성에 저장됨
        # model.parameters()로 접근할 수 있는 모델의 각 파라미터는 grad 속성을 가지는데, 여기에 저장됨
        optimizer.step() # 파라미터(가중치, 편향)의 grad 속성에 저장된 기울기를 발동시켜서, gradient descent기법으로 파라미터들을 변경(업데이트)시킴
        # max_vals, preds = torch.max(outputs, 1) 원래는 이건데, 첫 번째 반환값은 제일 높은 확률값, 두 번째 반환값은 그 제일 높은 확률값의 인덱스(정답번호)
        # torch.max(outputs, num)에서, num은 0이면 배치, 1이면 클래스를 말함, 그냥 1로 하면 되는 듯
        _, preds = torch.max(outputs, 1) # outputs는 모델이 예측한 값, 각 샘플에 대해 최대값의 인덱스 반환. 반환값은 (최대값, 그것의 인덱스)임
        running_loss += loss.item() * inputs.size(0) #
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset) * 100

    print(f'Epoch {str(epoch+1).zfill(2)}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f} %')
    
#시각화는 H,W,C가 일반적이고 딥러닝은 C,H,W가 일반적
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()  # matplotlib은 텐서형식을 처리하지 못함. 그래서 넘파이로 바꿔줘야됨
    # 근데 텐서에 .numpy()를 해줘도 차원순서가 (C,H,W)에서 (H,W,C)로 바뀌지는 않음 ToTensor()는 바꿈
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # 넘파이다운 차원순서로 바꿔주기

for data in test_loader: # test_loader는 배치값을 가지기에 한 번에 여러개의 사진을 수행함 전체 사진이 64개고, 배치가 8라면, test_loader는 8꾸러미 8개가 있는거임
    images, labels = data # 8꾸러미 중 한 꾸러미 수행(사진 8장) 이 코드에서는 배치=64임
    images = images.cuda()
    labels = labels.cuda()
    outputs, features = model(images) 
    # 그 8장에 대해 각각 모두 예측완료
    _, predicted = torch.max(outputs, 1) # torch.max의 두 번째 인자 1은 클래스 차원에서 최대값을 찾으라는 의미
    break # 이거 때매 반복안하고 가장 첫번째 배치단위만 predicted얻고 바로 끝냄
# 그렇기 때문에 그 한 배치에 대해(64장) heatmap 형성 반복문 수행함
classes =  ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
params = list(model.parameters())[-2] # 모델의 학습가능한 파라미터를 리스트로 만들고 거기서 뒤에서 2번째 파라미터를 불러옴
num = 0
for num in range(64):
    print(f"Item {str(num+1).zfill(2)}/64 | Ground Truth: {classes[int(labels[num])]} | Prediction: {classes[int(predicted[num])]}")
    
    # 집중영역 시각화, 위에서 고른 params(fc레이어 가중치)에서, 예측한 클래스(predicted[num])의 가중치 벡터를 가져옴
    # fc 레이어의 가중치는 단순 숫자인데, features는 제일 처음에 사진에서 공간적 요소를 캐치한 것이라서 이 둘을 가지고 시각화할 수 있음
    # fc 레이어가 classifier의 역할을 하기 때문에 예측은 얘가 하는 거임. 그래서 params로 불러온거임
    # 실제로 까보면 [-2](classifier인 fc 레이어의 가중치는 base.28.weight torch.Size([512, 512, 3, 3])로 나옴) & fc.weight torch.Size([10, 512]) 512개의 입력을 가지고 512개를 10가지 방식으로 조합해 계산한 결과 10개를 출력함
    # 0번 레이블 출력 값은 512개의 값을 어디에 계수를 크게 주고 어디를 작게 주고가 정해져있고, 1번 레이블 출력 값은 512개의 값 중 어디에 계수를 크게 주고 어디에 작게 주고가 정해져있음
    # 그렇게 입력된 512개의 출력을 가지고 얘를 0번 레이블(강아지)로 보는 행렬계수 세트를 사용해서 0 채널 값을 도출하고, 똑같이 1~9까지 모두 도출한 다음 softmax로 가장 큰 값을 예측값으로 선정함
    # 그럼 만약 모델이 예측값을 3번 레이블로 내렸고, 그게 cat이라면, 얘의 학습모델 내에서 10채널 중 3번 채널이 가지고 있는, 512개를 조합하는 행렬계수 세트를 불러오는 것이 params[int(predicted[num])]
    # 저 위에 모델 정의 내용을 보면, conv 레이어에만 다 통과시킨 상태를 features로써 반환함. 그것은 [512,H,W]일 것인데, 여기서는 [512,7,7]이라서 공간 차원을 49로 줄임. 3D->2D
    # params[int(predicted[num])]은 [10,512]이고, features[num].reshape(512,49)는 [512,49]이니까 행렬곱을 하면 [10,49]가 될 것임. 그걸 7 7로 리사이즈. 이걸 하면 heatmap 생성 가능
    # [512,7,7]일때부터 이미지 상에서 특성채널의 위치정보가 있었음. ex) 1,4 좌표에 둥글둥글함을 나타내는 채널값. 그렇다면 행렬곱을 한 결과물[10,7,7]에서도 어디가 0번 레이블이고 어디가 1번 레이블 등인지 위치정보가 있는 거임.
    overlay = params[int(predicted[num])].matmul(features[num].reshape(512,49)).reshape(7,7).cpu().data.numpy()
    # predicted[num]은, num이 반복문 이터레이터이기 때문에, (num)번째 사진에 대해 가장 확률이 높게 나온 인덱스(정답 번호)가 무엇인지 가리킴: 모델이 (num)번 문제에 적은 답
    #
    # features[num]은, (num)번째 사진에 대해, 이미지 특성 추출의 결과물을 가져오는 거임(flatten까지 하면서-> reshape)
    overlay = overlay - np.min(overlay)
    overlay = overlay / np.max(overlay) # heatmap 만들기, 정규화과정
    overlay_resized = skimage.transform.resize(overlay, [224, 224]) #원본 이미지와 같은 크기로 만들어서 겹쳐서 시각화하려고

    original_image = images[num].cpu()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    img = original_image.permute(1, 2, 0).numpy() # [C,H,W]인 이미지 텐서형식을 [H,W,C]로 만들어서 matplotlib할거임
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) #정규화, 원본 이미지이지만 matplotlib은 정규화해서 시각화하는게 일반적 + 정규화시킨 heatmap과 맞추기 위해
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(img)
    ax[1].imshow(overlay_resized, alpha=0.4, cmap='jet')
    ax[1].set_title("Learned Overlay")
    ax[1].axis('off')

    plt.show()