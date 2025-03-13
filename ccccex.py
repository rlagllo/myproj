import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#file = "C:\\Users\\김해창\\Desktop\\airfoildata\\2032c.dat"
#print(dats)

# with open(file, 'r') as f:
#     data = []
#     start_reading = False  # 데이터 시작 여부
#     for line in f:
#         print(line)
#         line = line.strip()  # 양쪽 공백 제거
#         if not line:  # 빈 줄은 건너뜁니다.
#             continue

#         # 데이터 시작을 확인하는 조건 (정확한 시작점을 찾는 조건으로 수정)
#         if not start_reading:
#             if int(line.split()[0]) >= 1.1:
#                 start_reading = True
#                 print(f"Data starts at line: {line}")  # 시작점 확인
#             continue  # 데이터 시작점 전까지는 건너뜁니다.

#         # 데이터 읽기
#         try:
#             values = [float(i) for i in line.split() if i]
#             if len(values) == 2:  # x, y 좌표가 정상적으로 읽혔을 경우
#                 values.append(0.0)  # z 좌표를 0으로 추가
#                 data.append(values)
#                 print(f"Line data: {values}")  # 데이터 확인
#         except ValueError:
#             continue  # 숫자가 아닌 값은 무시하고 넘어가기

# # 데이터 출력
# print(f"Collected data: {data[:5]}")  # 첫 5개의 데이터만 출력
            
# data = np.array(data)
# print(data)
# fname = os.path.basename(file) # ___.dat
# a1 = os.path.splitext(fname)[0] + '.csv'

# np.savetxt(a1, data, delimiter=',', header='x,y,z', comments='', fmt='%.6f')
# print(f"saved {a1}")
file = "C:\\Users\\김해창\\Desktop\\airfoildata\\whitcomb.dat"
with open(file, 'r') as f:
    next(f)
    #print('\n',file,'\n')
    data = []
    start_reading = False  # 데이터 시작 여부
    for line in f:
        #print(line)
        line = line.strip()  # 양쪽 공백 제거
        if not line:  # 빈 줄은 건너뜁니다.
            continue

        # 데이터 시작을 확인하는 조건 (정확한 시작점을 찾는 조건으로 수정)
        if not start_reading:
            #if line.split()[0] == '0.000000' or line.split()[0] == '1.00000' or line.split()[0] == '0.' or line.split()[0] == '0.0':
            if float(line.split()[0]) <= 1.1:
                start_reading = True
                #print(f"Data starts at line: {line}")  # 시작점 확인
            continue  # 데이터 시작점 전까지는 건너뜁니다.

        # 데이터 읽기
        try:
            values = [float(i) for i in line.split() if i]
            if len(values) == 2:  # x, y 좌표가 정상적으로 읽혔을 경우
                values.append(0.0)  # z 좌표를 0으로 추가
                data.append(values)
                #print(f"Line data: {values}")  # 데이터 확인
        except ValueError:
            continue  # 숫자가 아닌 값은 무시하고 넘어가기

# 데이터 출력
print(f"Collected data: {data[:5]}")  # 첫 5개의 데이터만 출력
            
data = np.array(data)
print("data =: ",data)
fname = os.path.basename(file) # ___.dat
a1 = os.path.splitext(fname)[0] + '.csv'

np.savetxt(a1, data, delimiter=',', header='x,y,z', comments='', fmt='%.6f')
print(f"saved {a1}")

x = data[:,0]
y = data[:,1]
z = data[:,2]
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')  # 3D 플롯을 위한 subplot 설정
#ax.scatter(x, y, z)

plt.plot(x,y,z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.xlim(0,1)  
plt.ylim(-0.3, 0.3)
plt.show()