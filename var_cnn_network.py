import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, batch_size):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.layer = nn.Sequential(
            # [배치사이즈, 채널, 너비, 높이] 지금 배치사이즈 ***  100  ***
            # 32-5
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5), # [150,3,32,32] -> [150,16,28,28]
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5), # [150,16,28,28] -> [150,32,24,24]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # [150,32,24,24] -> [150,32,12,12]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=6), # [150,32,12,12] -> [150,64,7,7]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2) # [150,64,7,7] -> [150,64,3,3]       
        )   
        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*3,200), # [150,64*3*3] -> [150,100]
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(200,100), 
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, 10) #[150,100] -> [150,10]
        )       
        
    def forward(self,x):
        out = self.layer(x)
        #out = out.view(self.batch_size,-1) # -> (150, remainder) layer(x)의 마지막 수행으로 나온 [150,64,3,3]를 가지고 1D 벡터로 변환
        out = out.view(out.size(0), -1) # 배치크기 동적할당
        out = self.fc_layer(out) #변환한 걸 fully connected layer에 투입
        
        return out
    
# 커넬을 써서 (예를들어 3x3) 영역의 feature를 수치화(?)함, 점점 사이즈가 줄어들것임.
# stride: 커넬이 한 번에 움직이는 픽셀 간격
# pooling: 주로 max pooling; 특정 영역에서 가장 큰 값으로 정보를 대표시킴, Con
# 이 과정이 하나의 Conv2d, 출력 채널은 인풋 1개에서 아웃풋은 16개 채널, 점점 늘어남,
# 마지막 Conv2d레이어의 결과물인 3D 텐서를 flatten을 통해 1차원 배열로 변환함: feature vector
# 이걸 가지고 fully connected layer에 넣음, 최종 출력은 Softmax/Argmax 등을 이용해서 가장 확률이 높은 값을 정답으로
# output size = (intput size(너비 높이 인듯듯) - kernel size + 2xpadding)/stride + 1: Conv2d 쓰면
# output size = (input size - kernel size)/stride + 1: maxpooling 쓰면