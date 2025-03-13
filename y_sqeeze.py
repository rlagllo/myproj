import torch
x = torch.tensor([[1,2,3],[[5, 10, 15]]])  # shape: (1, 1, 3)
print("x: ", x)
print(x.shape)  # torch.Size([1, 1, 3])
x_squeezed = x.squeeze() # 변수지정안하면 차원 크기 1인거 싹다 분해함
print(x_squeezed.shape)  # torch.Size([3])
print(x_squeezed)