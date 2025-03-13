import os
import glob
import numpy as np

var_path = "C:\\Users\\김해창\\Desktop\\VAR"
npz_path = "C:\\Users\\김해창\\Desktop\\VAR\\dataset"
path_to_train_2d_datasets = os.path.join(var_path, 'outputs', 'train', '*.csv')
train_2d_files = glob.glob(path_to_train_2d_datasets)

path_to_test_2d_datasets = os.path.join(var_path, 'outputs', 'test', '*.csv')
test_2d_files = glob.glob(path_to_test_2d_datasets)

train = np.empty((len(train_2d_files),28,28))
train_label = np.empty((len(train_2d_files)))

test = np.empty((len(test_2d_files),28,28))
test_label = np.empty((len(test_2d_files)))

for data_idx, data_path in enumerate(train_2d_files):
    filename = os.path.basename(data_path)
    train_label[data_idx] = int(filename.split('_')[0].replace('#',''))  # "#7-#0" -> 7
    csv_data = np.loadtxt(data_path,delimiter=',')
    for i in range(28):
        for j in range(28):
            train[data_idx,i,j] = csv_data[i][j]

for data_idx, data_path in enumerate(test_2d_files):
    filename = os.path.basename(data_path)
    test_label[data_idx] = int(filename.split('_')[0].replace('#',''))
    csv_data = np.loadtxt(data_path,delimiter=',')
    for i in range(28):
        for j in range(28):
            test[data_idx,i,j] = csv_data[i][j]

data = np.concatenate((train, test), axis=0)
label = np.concatenate((train_label, test_label), axis=0)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
label = label[indices]

train_size = int(len(data) * 0.7)
valid_size = int(len(data) * 0.2)

train, valid, test = np.split(data, [train_size, train_size + valid_size])
train_label, valid_label, test_label = np.split(label, [train_size, train_size + valid_size])

train_x_path = os.path.join(var_path, "dataset", "train_x.npz")
train_y_path = os.path.join(var_path, "dataset", "train_y.npz")

valid_x_path = os.path.join(var_path, "dataset", "valid_x.npz")
valid_y_path = os.path.join(var_path, "dataset", "valid_y.npz")

test_x_path = os.path.join(var_path, "dataset", "test_x.npz")
test_y_path = os.path.join(var_path, "dataset", "test_y.npz")


np.savez(train_x_path, train_x=train)  #훈련 데이터
np.savez(train_y_path, train_y=train_label) #훈련 데이터 답지(레이블)

np.savez(valid_x_path, valid_x=valid)
np.savez(valid_y_path, valid_y=valid_label)

np.savez(test_x_path, test_x=test)
np.savez(test_y_path, test_y=test_label)