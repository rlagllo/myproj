import os, glob
import numpy as np
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

narutopath = "C:\\Users\\김해창\\Desktop\\VAR\\naruto\\naruto"
sasukepath = "C:\\Users\\김해창\\Desktop\\VAR\\naruto\\sasuke"
npz_path = "C:\\Users\\김해창\\Desktop\\VAR\\naruto\\npz"
transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class DataLoader():
    def __init__(self, dataset_name = 'naruto_', shuffle=False,npz_path = npz_path, train_size = 0.7, valid_size = 0.2):
        # Initialize variables
        self.npz_path = npz_path
        self.train_size = train_size
        self.valid_size = valid_size
        self.narutos = glob.glob(os.path.join(narutopath,'*.jpg'))
        self.sasukes = glob.glob(os.path.join(sasukepath,'*.jpg'))
        
        self.narutox=np.empty((len(self.narutos),256,256,3))
        self.narutoy=np.empty(len(self.narutos))
        
        self.sasukex=np.empty((len(self.sasukes),256,256,3))
        self.sasukey=np.empty(len(self.sasukes))

        
        self.dataset_name = dataset_name
        self.shuffle = shuffle
        self.load_x_y()
        
        self.x = np.concatenate((self.narutox, self.sasukex), axis=0)
        self.y = np.concatenate((self.narutoy, self.sasukey), axis=0)
        if self.shuffle:
            indices = np.arange(data.shape[0])  
            np.random.shuffle(indices)
            data = data[indices]
            label = label[indices]
        self.split_data_save()
    
    def load_x_y(self):
        for data_idx, data_path in enumerate(self.narutos): #저장한 나루토사진들의 넘버링과 주소
            image = cv2.imdecode(np.fromfile(data_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            image = cv2.resize(image, (256, 256))
            image = (image / 127.5) - 1
            self.narutox[data_idx] = image
            self.narutoy[data_idx] = 0

        for data_idx, data_path in enumerate(self.sasukes):
            image = cv2.imdecode(np.fromfile(data_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            image = cv2.resize(image, (256, 256))
            image = (image / 127.5) - 1
            self.sasukex[data_idx] = image
            self.sasukey[data_idx] = 1
            
    def split_data_save(self):    
        data_size = len(self.x)
        train_size = int(data_size * self.train_size)
        valid_size = int(data_size * self.valid_size)
        train, valid, test = np.split(self.x, [train_size, train_size + valid_size])
        train_label, valid_label, test_label = np.split(self.y, [train_size, train_size + valid_size])
        
        train_x_path = os.path.join(self.npz_path, f"train_{self.dataset_name}x.npz")
        train_y_path = os.path.join(self.npz_path, f"train_{self.dataset_name}y.npz")

        valid_x_path = os.path.join(self.npz_path, f"valid_{self.dataset_name}x.npz")
        valid_y_path = os.path.join(self.npz_path, f"valid_{self.dataset_name}y.npz")

        test_x_path = os.path.join(self.npz_path, f"test_{self.dataset_name}x.npz")
        test_y_path = os.path.join(self.npz_path, f"test_{self.dataset_name}y.npz")

        np.savez(train_x_path, train_x=train)  #훈련 데이터
        np.savez(train_y_path, train_y=train_label) #훈련 데이터 답지(레이블)

        np.savez(valid_x_path, valid_x=valid)
        np.savez(valid_y_path, valid_y=valid_label)

        np.savez(test_x_path, test_x=test)
        np.savez(test_y_path, test_y=test_label)
        
    def load(self):
        train_x_path = os.path.join(npz_path, "train_naruto_x.npz")
        train_y_path = os.path.join(npz_path, "train_naruto_y.npz")

        train_x_data = np.load(train_x_path)['train_x']  # 'train_x' 키로 데이터 로드
        train_x_data = np.load(train_y_path)['train_y']  # 'train_y' 키로 데이터 로드

        # 유효성 검증 데이터 로드
        valid_x_path = os.path.join(npz_path, "valid_naruto_x.npz")
        valid_y_path = os.path.join(npz_path, "valid_naruto_y.npz")

        valid_x_data = np.load(valid_x_path)['valid_x']
        valid_y_data = np.load(valid_y_path)['valid_y']

        # 테스트 데이터 로드
        test_x_path = os.path.join(npz_path, "test_naruto_x.npz")
        test_y_path = os.path.join(npz_path, "test_naruto_y.npz")

        test_x_data = np.load(test_x_path)['test_x']
        test_y_data = np.load(test_y_path)['test_y']

        return train_x_data, train_x_data, valid_x_data, valid_y_data, test_x_data, test_y_data
if __name__ == "__main__":
    dl=DataLoader()
    dl.split_data_save()
    dl.load()
    print("끝")