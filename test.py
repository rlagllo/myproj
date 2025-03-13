# Import necessary libraries
import numpy as np
import pandas as pd
import math
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import defaultdict
os.chdir("C:\\Users\\김해창\\Desktop\\.venv")
# Load vertices and faces data
vertices = pd.read_csv('vertices.csv', header=None).values  # [x, y, z]
faces = pd.read_csv('faces.csv', header=None).values  # [v1, v2, v3]

# 법선 벡터 계산, 점3개(삼각형)을 가져와 법선벡터계산, 단위벡터로 정규화
def calculate_face_normals(vertices, faces):
    normals = []
    for face in faces:
        v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        # Compute vectors for two edges of the triangle
        edge1 = v2 - v1
        edge2 = v3 - v1
        # Calculate the cross product of the edges to get the face normal
        normal = np.cross(edge1, edge2)
        # Normalize the normal vector
        normal /= np.linalg.norm(normal)
        normals.append(normal)
    return np.array(normals)

face_normals = calculate_face_normals(vertices, faces)


# 이번엔 각 점으로부터 만들어진 모든 단위벡터를 누적하고 평균계산, 다시 정규화
vertex_normals = np.zeros(vertices.shape)
counts = np.zeros(len(vertices))

for i, face in enumerate(faces):
    for vertex_idx in face:
        vertex_normals[vertex_idx] += face_normals[i]
        counts[vertex_idx] += 1
vertex_normals = (vertex_normals.T / counts).T
vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]


# Estimate mean curvature using dot product of vertex position and normal
# Mean curvature approximation: H ~ dot(vertex normal, vertex position)
# 점에서 법선벡터와 위치벡터간의 내적
curvatures = np.einsum('ij,ij->i', vertex_normals, vertices)

# Calculate the mean point as the estimated head center based on curvature
center_point = np.average(vertices, axis=0, weights=np.abs(curvatures))

# Output the calculated center point for debugging
print("center_point 완료",center_point)

def find_boundary_edges(faces):
    edge_count = defaultdict(int)

    # 모든 삼각형 면에서 에지를 추출하고 카운트
    for face in faces:
        edges = [
            tuple(sorted([face[0], face[1]])),  # (v1, v2)
            tuple(sorted([face[1], face[2]])),  # (v2, v3)
            tuple(sorted([face[2], face[0]])),  # (v3, v1)
        ]
        for edge in edges:
            edge_count[edge] += 1

    # 한 번만 나타난 에지 (경계 에지)
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    return boundary_edges

# 경계점 찾기
def find_boundary_vertices(boundary_edges):
    boundary_vertices = set()
    for edge in boundary_edges:
        boundary_vertices.update(edge)  # 경계 에지의 두 정점을 추가
    return list(boundary_vertices)

# 1. 경계 에지 추출
boundary_edges = find_boundary_edges(faces)

# 2. 경계점 추출
boundary_vertices_indices = find_boundary_vertices(boundary_edges)

# 3. 경계점 좌표
boundary_vertices = vertices[boundary_vertices_indices]

# 출력 (경계점 확인)
print(f"경계점 개수: {len(boundary_vertices)}, 경계점 따기 완료")
# print(f"경계점 좌표:\n{boundary_vertices}")

# 상위 몇 퍼센트의 z 값을 가진 경계점 사용할지 설정 (예: 상위 10%)
top_percentage = 1  # 상위 10% 선택
num_top_points = int(len(boundary_vertices) * top_percentage)

# z 값을 기준으로 정렬 (내림차순)
sorted_boundary_vertices = boundary_vertices[np.argsort(boundary_vertices[:, 2])[::-1]]

# 상위 num_top_points 개의 점 선택
top_z_boundary_vertices = sorted_boundary_vertices[:num_top_points]

# 선택된 점들의 평균으로 새로운 중심점 계산
new_center_point = np.mean(top_z_boundary_vertices, axis=0)
print("새로운 중심점 (상위 z 값 평균):", new_center_point)

# # 3D 플롯 생성
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# sample_ratio = 0.05  # 조금만 샘플링
# num_samples = int(len(vertices) * sample_ratio)
# sample_indices = np.random.choice(len(vertices), num_samples, replace=False)
# sample_vertices = vertices[sample_indices]

# # 경계점 플롯
# ax.scatter(center_point[0], center_point[1], center_point[2],
#            c='y', marker='d', s=100, label='Center Point new')
# ax.scatter(boundary_vertices[:, 0], boundary_vertices[:, 1], boundary_vertices[:, 2],
#            c='r', marker='o', label='Boundary Vertices')
# ax.scatter(new_center_point[0], new_center_point[1], new_center_point[2],
#            c='b', marker='*', s=100, label='Center Point')
# ax.scatter(sample_vertices[:, 0], sample_vertices[:, 1], sample_vertices[:, 2],
#            c='gray', marker='.', alpha=0.5, label='Sampled Vertices')
# # 레이블과 제목 추가
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')
# ax.set_title('Boundary Vertices Plot')
# ax.legend()
# plt.show()

#  center_point Gs
# new cen_point C
# boundary_vertices A




k = 2   #factor
step = 15
vector1 = np.array([0, 0, 0])
vector1 = new_center_point - center_point
m = np.linalg.norm(vector1)
vector1_unit = vector1 / m
#G'
center_point_prime = new_center_point + vector1 * k

point_Ps = np.empty((0,3))

for point_A in boundary_vertices:
    vector2 = np.array([0, 0, 0])
    vector2 = new_center_point - point_A
    l = np.linalg.norm(vector2)
    vector2_unit = vector2 / l

    theta_rng = np.linspace(0, 90, step)

    for theta in theta_rng:
        theta_rad = np.radians(theta)
        point_P = point_A + l*(1-math.cos(theta_rad))*vector2_unit + m*k*math.sin(theta_rad)
        point_Ps = np.vstack((point_Ps, point_P))
print("섹스")
print(len(point_Ps))

sample_ratio = 0.1  # 조금만 샘플링
num_samples = int(len(vertices) * sample_ratio)
sample_indices = np.random.choice(len(vertices), num_samples, replace=False)
sample_vertices = vertices[sample_indices]

# 3D 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 경계점 플롯
ax.scatter(point_Ps[:,0], point_Ps[:,1], point_Ps[:,2],
          c='y', marker='d', s=100, label='point_Ps')
ax.scatter(center_point[0], center_point[1], center_point[2],
           c='r', marker='+', s=100, label='Center Point new')
ax.scatter(new_center_point[0], new_center_point[1], new_center_point[2],
           c='b', marker='*', s=100, label='Center Point')
ax.scatter(sample_vertices[:, 0], sample_vertices[:, 1], sample_vertices[:, 2],
           c='gray', marker='.', alpha=0.5, label='Sampled Vertices')



# 레이블과 제목 추가
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Boundary Vertices Plot')
ax.legend()
plt.show()

