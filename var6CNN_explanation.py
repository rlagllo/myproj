# VGG16의 특징 추출 부분
# CNN, convolution과정을 통해 geometric feature를 점차 수치화하여 추출
# 입력 채널 (in_channels=3): 입력 이미지의 채널 수. 여기서는 3채널 이미지(예: RGB 이미지)
# 출력 채널 (out_channels=16): 이 층에서 생성할 특징 맵의 채널 수. 여기서는 16개의 필터를 사용하여 특징을 추출
# 커널 크기 (kernel_size=5): 필터의 크기입니다. 이 경우 5x5 크기의 필터가 사용됩니다. 커널을 이미지에 적용하여 특징을 추출
# 출력 크기: 입력 이미지 크기가 [150, 3, 32, 32]였을 때, 필터 크기 5x5로 컨볼루션을 수행하면 출력 크기는 [150, 16, 28, 28] (채널 수는 16으로 늘어나고, 너비와 높이는 작아짐)

features = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(), # 비선형성 추가
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # ... (이런 식으로 여러 층들이 이어짐)
)

# output size = (intput size(너비 높이 인듯듯) - kernel size + 2xpadding)/stride + 1: Conv2d 쓰면
# output size = (input size - kernel size)/stride + 1: maxpooling 쓰면

# 분류 부분 (완전 연결 층)
classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Linear(4096, 10)  # 예시로 10개의 클래스를 분류
)
# 이게 vgg16의 내부구조임. 
# self.base = base_model.features 이건 저 위쪽 한 덩어리만 가져오겠다는 의미
# 그렇게 했을 때 self.base.parameters()는
# 모든 컨볼루션 계층(Conv2d, ReLU, MaxPool2d 등)의 가중치(weights)와 편향(bias)을 말함
# nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5) 여기서, 16개의 5x5필터와 각 필터에 대한 1개의 편향이 학습될 것임. 이게 parameter임.

# self.layer = nn.Sequential(
#     # [배치사이즈, 채널, 너비, 높이] 지금 배치사이즈 ***  100  ***
#     # 32-5
#     nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5), # [150,3,32,32] -> [150,16,28,28]
#     nn.ReLU(),
#     nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5), # [150,16,28,28] -> [150,32,24,24]
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2,stride=2), # [150,32,24,24] -> [150,32,12,12]
#     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=6), # [150,32,12,12] -> [150,64,7,7]
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3,stride=2) # [150,64,7,7] -> [150,64,3,3]       
# )   
# 그럼 150개의 이미지가 들어오는데, 각 이미지마다 5x5의 커넬로 순회
# 사이즈가 32x32에서 28x28로 줄어든 것은 커넬에 의한 것
# 아웃풋은 RGB채널이 아니고 16개의 수치화된 특징