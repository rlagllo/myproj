import torch

# 2x3 크기의 텐서를 생성합니다.
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("x.shape =", x.shape,'\n') # 2x3 행렬
# dim=0으로 gather
# 첫번째 인덱스는 [1, 2, 3]이고, 두번째 인덱스는 [4, 5, 6]입니다.
# 따라서 idx가 0이면 [1, 2, 3], 1이면 [4, 5, 6]이 반환됩니다.

# [0,1]을 unsqeeze(1)을하면 

print("[0,1] shape= ", torch.tensor([0,1]).shape)
print(torch.tensor([0,1]))
print()
print("[0,1].unsqeeze(0) shape= ", torch.tensor([0,1]).unsqueeze(0).shape)
print(torch.tensor([0,1]).unsqueeze(0))
print()
print("[0,1].unsqeeze(1) shape= ", torch.tensor([0,1]).unsqueeze(1).shape)
print(torch.tensor([0,1]).unsqueeze(1))
print()

print("x=", x)
idx = torch.tensor([0, 1]).unsqueeze(1).repeat(1,2)
idx = torch.tensor([[0,1],[1,0]])
print("idx=", idx, '\n')
# tensor([[0, 0, 0],[1, 1, 1]])
# torch.Size([2, 3]) 이건 2행 3열인데 그냥 torch.Size([2]) 이건 1차원 벡터의 요소가 2개인거
result = x.gather(0, idx) # idx에 있는 값을 가지고 x에서 값을 가져오는 연산
# gather(dim, index): dim=0: 행방향(세로) dim=1: 열방향(가로)로 인덱싱
# [0]번째 애 선택, 
print(result)
# tensor([[1, 2, 3],[4, 5, 6]])