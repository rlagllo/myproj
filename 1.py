import torch


print(torch.__version__)  # PyTorch 버전 확인
print(torch.cuda.is_available())  # CUDA 사용 가능 여부 확인

print(torch.cuda.is_available())  # CUDA 사용 가능 여부 확인
print(torch.cuda.get_device_name(0))  # 사용 가능한 GPU 이름 출력 (있다면)

