import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
import pickle
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict



os.chdir("C:\\Users\\김해창\\Desktop\\cifar-10-batches-py")
X_train, y_train = [],[] # 학습용, X: 이미지, y: 레이블
batch_path = "data*"
batches = glob.glob(batch_path)

for single_batch in batches: # 여기서 batch는 넘파이고, X_train은 리스트임. 그래서 리스트 안에 5개의 넘파이가 들어있는 것이기 때문에 후에 vstack이나 concat을 해야함, 반면에 y(레이블)들은 리스트임
    print(single_batch)
    batch = unpickle(single_batch)
    print(batch.keys())
    #print(batch[b'data'])
    X_train.append(batch[b'data'])
    y_train.append(batch[b'labels'])

    # 레이블 데이터 (y_train의 원소)
    print("batch[b'labels'] 타입:", type(batch[b'labels']))  # 리스트인지 확인. 리스트임
    print("batch[b'labels'] 길이:", len(batch[b'labels']))  # 데이터 개수 확인
    print("batch[b'labels'] 첫 10개:", batch[b'labels'][:10])  # 일부 데이터 출력
    
#print("X_train len : ", len(X_train))
X_train = np.vstack(X_train).reshape(-1,3,32,32).astype(np.uint8) # 넘파이: N,C,H,W     N: 데이터/배치 개수
y_train = np.hstack(y_train).reshape(-1).astype(np.int64)

test_path = "C:\\Users\\김해창\\Desktop\\cifar-10-batches-py\\test_batch"

test_batch = unpickle(test_path)

X_test = test_batch[b'data'].reshape(-1,3,32,32).astype(np.uint8)
y_test = np.array(test_batch[b'labels']).astype(np.int64) # 애초에 리스트라서 넘파이로 명시적으로 바꾸기

train_size = int(len(X_train) * 0.9)

train_data, valid_data = np.split(X_train, [train_size])
train_label, valid_label =np.split(y_train,[train_size])

proj_path = "C:\\Users\\김해창\Desktop\\cifar-10-batches-py"

train_x_path = os.path.join(proj_path, "refined", "train_x.npz")
train_y_path = os.path.join(proj_path, "refined", "train_y.npz")

valid_x_path = os.path.join(proj_path, "refined", "valid_x.npz")
valid_y_path = os.path.join(proj_path, "refined", "valid_y.npz")

test_x_path = os.path.join(proj_path, "refined", "test_x.npz")
test_y_path = os.path.join(proj_path, "refined", "test_y.npz")

np.savez(train_x_path, train_x=train_data)  #훈련 데이터
np.savez(train_y_path, train_y=train_label) #훈련 데이터 답지(레이블)

np.savez(valid_x_path, valid_x=valid_data)
np.savez(valid_y_path, valid_y=valid_label)

np.savez(test_x_path, test_x=X_test)
np.savez(test_y_path, test_y=y_test)


# first_image = batch[b'data'][0]
# print("first_image.shape: ", first_image.shape)
# first_image = first_image.reshape(3,32,32)
# print("first_image.reshape shape: ", first_image.shape)  # C,W,H
# first_image = first_image.transpose(1,2,0) # 숫자는 순서를 의미함, 저 순서대로 차원바꾸기 C,W,H -> H,W,C . 0번째인 C가 괄호안의 3번쨰로 가니까 C는 3번쨰 채널로 감

# plt.imshow(first_image)
# plt.axis("off")
# plt.show()
