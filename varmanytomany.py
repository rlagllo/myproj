import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_length = 60
epochs = 15
hidden_size = 50
num_layers = 2
output_size = 1 # many to one:1
batch_size = 1
lr = 0.0001
ticker = "SBUX"

"""
Step1: Preprocess Datasets
"""
data = yf.download(ticker, start="2020-01-01", end="2023-12-31") # ticker: 종목 코드, 기간
data = data['Close'].values.reshape(-1, 1) # close: 종가, reshape(-1,1) -1: 행 개수 자동, 열 개수 1개
input_size = data.shape[1] # data의 열 개수를 input_size에 저장

scaler = MinMaxScaler(feature_range=(0, 1)) # 데이터를 0,1로 정규화하는 scalar지정
data_normalized = scaler.fit_transform(data) # data에 scalar적용

def create_sequences(data, seq_length): # 함수 정의
    sequences = [] # 여기에 저장될 거임 훈련
    targets = [] # 여기에 저장될 거임 예측대상
    for i in range(len(data) - seq_length): # 데이터 돌면서 sequence랑 target에 나눠서 저장
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+1:i+seq_length+1]) # many to one: targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets) # 저장한 거 넘파이로 변환

X, y = create_sequences(data_normalized, seq_length) # 위에서 저장한 걸 x,y에 실행

# X,y는 sequence, target. X는 훈련용, y는 예측할 애들. X,y를 똑같은 크기로 먼저 나눈 후에에 TensorDataset에서 합치는 거

# 만약 데이터가 [10, 20, 30, 40, 50, 60]이고 seq_length = 3이라면:
# X[0]=[10,20,30],y[0]=40, X[1]=[20,30,40],y[1]=50, X[2]=[30,40,50],y[2]=60

split = int(0.8 * len(X)) # 80% 학습, 20% 테스트
X_train, X_test = X[:split], X[split:] #나누기
y_train, y_test = y[:split], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32) # 넘파이를 텐서로 변환
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train) # pytorch에서 주는 TensorDataset 객체
test_dataset = TensorDataset(X_test, y_test) # 넣어준 (나눠진)X,y를 하나로 묶어줌, dataset[0]=(X[0],y[0])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#Step2: Define LSTM Network

# LSTM은 시점별 출력을 봐야됨. out = out[:, -1, :]: 이건 시퀀스의 마지막 시점의 출력을 선택함
class LSTMModel(nn.Module): # (self, 위에서 정의한 애들)
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, output_size)
    
    def forward(self, x):
        hidden_cell = (torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device),
                            torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))
    
        out, _ = self.lstm(x, hidden_cell) # self.lstm은 반환값이 2개임, 여기서 2번째 거 무시하려고 _ 사용.
        out = out.contiguous().view(-1, self.hidden_size) # 불연속적인 텐서를 연속적으로 만듦, 모든 시점을 1D로 펼치고, 레이어에 통과시킬것
        #out = out[:, -1, :]
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.view(x.size(0), x.size(1), self.fc2.out_features) #
        return out

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
for epoch in range(epochs):
    train_loss = 0.0
    for sequences, targets in train_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    print(f'Epoch {str(epoch+1).zfill(2)}/{epochs}, Loss: {train_loss/len(train_loader):.7f}')

"""
Step4: Evaluate model
"""
model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for sequences, targets in test_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        output = model(sequences)
        predictions.extend(output.squeeze().tolist())
        actuals.extend(targets.squeeze().tolist())        
        
        # if output.size(0) == 1:
        #     predictions.append(outputs.detach().cpu().numpy())
        #     actuals.append(targets.item())
        # else:
        #     predictions.extend(output.squeeze().tolist())
        #     actuals.extend(targets.squeeze().tolist())

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual')
plt.show()
plt.close()
plt.plot(actuals, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()