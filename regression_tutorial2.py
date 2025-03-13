
import os
os.chdir("C:\\Users\\김해창\\Desktop\\.venv")

import os
import torch
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

Draglift = np.genfromtxt('DragLift.csv', delimiter=',', skip_header = 1, dtype=np.float32)
print(Draglift, Draglift.shape)

Draglift_transpose = np.transpose(Draglift)
print(Draglift_transpose, Draglift_transpose.shape)

plt.figure(figsize = (10,8))
plt.scatter(Draglift_transpose[0], Draglift_transpose[1], c = 'tomato',
            linewidth = 3, label='Drag coefficient')
plt.legend(fontsize = 12)
plt.ylabel('Drag coefficient', fontsize = 15)
plt.xlabel('Velocity ($u$)', fontsize = 15)
plt.show()

plt.figure(figsize = (10,8))
plt.scatter(Draglift_transpose[0], Draglift_transpose[2],
            c = 'mediumblue', linewidth = 3, label='Lift coefficient')
plt.legend(fontsize = 12)
plt.ylabel('Lift coefficient', fontsize = 15)
plt.xlabel('Velocity ($u$)', fontsize = 15)
plt.show()

np.random.seed(42)
#위 플랏에서 2,6,9번째 데이터를 학습데이터로 쓰겠다
test_indices = [2, 6, 9]

print(test_indices)

# Split the array into train and test sets
train_data = np.delete(Draglift, test_indices, axis=0)
test_data = Draglift[test_indices]

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

#Model architecture
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(1, 64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        x = self.linear3(x)
        return (x)

#신경망을 연산장치에 업로드
model = NN().to(device)
#Optimizer 및 learning rate 설정
optimizer = Adam(model.parameters(), lr=0.0002)
#loss function 설정
mse= torch.nn.MSELoss()

train_x = train_data[:, :1, ...]
train_y = train_data[:, 1:, ...]
test_x = test_data[:, :1, ...]
test_y = test_data[:, 1:, ...]

train_x_torch = torch.from_numpy(train_x)
train_y_torch = torch.from_numpy(train_y)
test_x_torch = torch.from_numpy(test_x)
test_y_torch = torch.from_numpy(test_y)

print(train_x_torch.shape)
print(train_y_torch.shape)
print(test_x_torch.shape)
print(test_y_torch.shape)
#[8,1]: 1개짜리 데이터가 8개 있다.

#Monitor training
train_loss_metric = []
test_loss_metric = []

#learning rate= lr/epoch ??

num_epochs=10000
for epoch in range (1, num_epochs+1):
    model.train()
    optimizer.zero_grad()

    input = train_x_torch.to(device)
    target = train_y_torch.to(device)


    gen_output = model(input)

    loss = mse(gen_output, target)

    loss.backward()

    train_loss_metric.append(loss.detach().cpu().numpy())

    optimizer.step()

    model.eval() #모델 학습 잠그기
    with torch.autograd.no_grad():
        input = test_x_torch.to(device)
        target = test_y_torch.to(device)
        gen_output = model(input)
        loss = mse(gen_output, target)
        test_loss_metric.append(loss.detach().cpu().numpy())


    if epoch % 4000 ==0:

        plt.figure()
        plt.plot(list(range(len(train_loss_metric))),train_loss_metric, label = "train_loss")
        plt.plot(list(range(len(test_loss_metric))),test_loss_metric, label = "test_loss")
        plt.ylim(0,0.001)
        plt.title('MSE')
        plt.show()

model.eval() #모델 학습 잠그기
with torch.autograd.no_grad():
    input = test_x_torch.to(device)
    gen_output = model(input)
    gen_output = gen_output.detach().cpu().numpy()

print(gen_output)

plt.figure(figsize = (10,8))
plt.scatter(Draglift_transpose[0], Draglift_transpose[1], c = 'tomato',
            linewidth = 3, label='True data')

plt.scatter(np.transpose(test_x)[0], np.transpose(gen_output)[0], c = 'mediumblue',
            linewidth = 3, label='Predicted data')

plt.legend(fontsize = 12)
plt.ylabel('Drag coefficient', fontsize = 15)
plt.xlabel('Velocity ($u$)', fontsize = 15)
plt.show()

plt.figure(figsize = (10,8))
plt.scatter(Draglift_transpose[0], Draglift_transpose[2], c = 'tomato',
            linewidth = 3, label='True data')

plt.scatter(np.transpose(test_x)[0], np.transpose(gen_output)[1], c = 'mediumblue',
            linewidth = 3, label='Predicted data')

plt.legend(fontsize = 12)
plt.ylabel('Lift coefficient', fontsize = 15)
plt.xlabel('Velocity ($u$)', fontsize = 15)
plt.show()