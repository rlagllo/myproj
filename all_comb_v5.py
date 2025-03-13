import os
import time
import glob
import subprocess
import numpy as np
from stl import mesh
import pandas as pd
from collections import defaultdict
import open3d as o3d
import matplotlib.pyplot as plt
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

def compute_edge_length(vertices, face_indices):
    edge_lengths = []
    for face in face_indices:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        edge_lengths.extend([
            np.linalg.norm(v0 - v1),
            np.linalg.norm(v1 - v2),
            np.linalg.norm(v2 - v0)
        ])
    return np.mean(edge_lengths)

os.chdir("C:\\Users\\김해창\\Desktop\\mesh_extraction_tool_")
image_folder_path = "C:\\Users\\김해창\\Desktop\\mesh_extraction_tool_"
exe_path = "C:\\Users\\김해창\\Desktop\\mesh_extraction_tool_\\mesh_extraction_tool.exe"
output_path = "C:\\Users\\김해창\\Desktop\\mesh_extraction_tool_"
output_folder_path = "C:\\Users\\김해창\\Desktop\\mesh_extraction_tool_\\output_folder"
new_vertices_path = "C:\\Users\\김해창\\Desktop\\mesh_extraction_tool_\\new_vertices.csv"
vertices_path = "C:\\Users\\김해창\\Desktop\\mesh_extraction_tool_\\vertices.csv"
faces_path = "C:\\Users\\김해창\\Desktop\\mesh_extraction_tool_\\faces.csv"
vertices_b_path = "C:\\Users\\김해창\\Desktop\\mesh_extraction_tool_\\vertices_b.csv"
faces_b_path = "C:\\Users\\김해창\\Desktop\\mesh_extraction_tool_\\faces_b.csv"
merged_vertices_path = "C:\\Users\\김해창\\Desktop\\mesh_extraction_tool_\\merged_vertices.csv"
merged_faces_path = "C:\\Users\\김해창\\Desktop\\mesh_extraction_tool_\\merged_faces.csv"
output_b_stl_path = "C:\\Users\\김해창\\Desktop\\mesh_extraction_tool_\\output_b.stl"
#이미지들 전부 수집
image_files = sorted(glob.glob(os.path.join(image_folder_path,"*.png")))

for i, image_path in enumerate(image_files):
    timeout = 20
    now_img_it = i
    try:
        original_name = os.path.basename(image_path)
        input_image_path = os.path.join(image_folder_path, "input.png")
    
        os.rename(image_path, input_image_path)
        # exe 실행
        process = subprocess.Popen([exe_path], shell=True)
        print("툴 실행")

        # 실행 대기 (timeout)
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                print(f"타임아웃: {image_path}")
                break

            # csv 파일 생성 확인
            csv_files = glob.glob(os.path.join(image_folder_path, "*.csv"))
            if csv_files:
                print(f"생성 성공: {csv_files}")
                break
            time.sleep(1)
        time.sleep(1)
        # Load vertices and faces data
        vertices = pd.read_csv('vertices.csv', header=None, skiprows=1).values  # [x, y, z]
        faces = pd.read_csv('faces.csv', header=None, skiprows=1).values  # [v1, v2, v3]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x=vertices[:,0]
        y=vertices[:,1]
        z=vertices[:,2]


        ax.scatter(x,y,z,c='r',marker='o')
        ax.set_ylabel('all vertices')
        # 플롯 표시
        plt.show()
        face_normals = calculate_face_normals(vertices, faces)

        center_point = vertices.mean()

        # # 이번엔 각 점으로부터 만들어진 모든 단위벡터를 누적하고 평균계산, 다시 정규화
        # vertex_normals = np.zeros(vertices.shape)
        # counts = np.zeros(len(vertices))

        # for i, face in enumerate(faces):
        #     for vertex_idx in face:
        #         vertex_normals[vertex_idx] += face_normals[i]
        #         counts[vertex_idx] += 1
        # vertex_normals = (vertex_normals.T / counts).T
        # vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]


        # # Estimate mean curvature using dot product of vertex position and normal
        # # Mean curvature approximation: H ~ dot(vertex normal, vertex position)
        # # 점에서 법선벡터와 위치벡터간의 내적
        # curvatures = np.einsum('ij,ij->i', vertex_normals, vertices)

        # # Calculate the mean point as the estimated head center based on curvature
        # center_point = np.average(vertices, axis=0, weights=np.abs(curvatures))

        # Output the calculated center point for debugging
        print("center_point 완료",center_point)

        # 경계 에지 추출
        boundary_edges = find_boundary_edges(faces)

        # 경계점 추출
        boundary_vertices_indices = find_boundary_vertices(boundary_edges)

        # 경계점 좌표
        boundary_vertices = vertices[boundary_vertices_indices]

        # 경계점 좌표 (boundary_vertices가 주어졌다고 가정)

        # 출력 (경계점 확인)
        print(f"경계점 개수: {len(boundary_vertices)}, 경계점 따기&추가 완료")

        # 상위 몇 퍼센트의 z 값을 가진 경계점 사용할지 설정 (예: 상위 10%)
        top_percentage = 1  # 상위 % 선택
        num_top_points = int(len(boundary_vertices) * top_percentage)

        # z 값을 기준으로 정렬 (내림차순)
        sorted_boundary_vertices = boundary_vertices[np.argsort(boundary_vertices[:, 2])[::-1]]

        # 상위 num_top_points 개의 점 선택
        top_z_boundary_vertices = sorted_boundary_vertices[:num_top_points]

        # 선택된 점들의 평균으로 새로운 중심점 계산
        new_center_point = np.mean(top_z_boundary_vertices, axis=0)
        print("새로운 중심점 (상위 z 값 평균):", new_center_point)
        import numpy as np

        A = center_point
        B = new_center_point
        # 점 A, B, C 좌표

        # 벡터 AB, BC 정의
        #AB = A - B
        AB = B-A
        #결과: positions에 저장된 점들을 사용해 뒷통수 메쉬 생성 가능
        #각도를 점진적으로 증가시켜 180도에 도달하는 방식
        angles = np.linspace(0, np.pi, 50)  # 0에서 180도(π 라디안)까지 100 단계로 나눔
        positions = []
        
        for j,C in enumerate(boundary_vertices):
            # 처리할 작업
            #target_length = 150
            original_BC_length = np.linalg.norm(C-B)
            #length_step = np.linspace(original_BC_length,target_length,len(angles))
            # 회전할 축 (AB와 직각인 축 필요)
            rotation_axis = np.cross(AB, C-B).astype(float)  # float로 변환
            rotation_axis /= np.linalg.norm(rotation_axis)  # 단위 벡터로 정규화
        
            for i, theta in enumerate(angles):
                # 회전 행렬 생성 (로드리게스 회전 공식)
                #current_length = length_step[i]
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                u_x, u_y, u_z = rotation_axis

                rotation_matrix = np.array([
                    [cos_theta + u_x ** 2 * (1 - cos_theta), u_x * u_y * (1 - cos_theta) - u_z * sin_theta, u_x * u_z * (1 - cos_theta) + u_y * sin_theta],
                    [u_y * u_x * (1 - cos_theta) + u_z * sin_theta, cos_theta + u_y ** 2 * (1 - cos_theta), u_y * u_z * (1 - cos_theta) - u_x * sin_theta],
                    [u_z * u_x * (1 - cos_theta) - u_y * sin_theta, u_z * u_y * (1 - cos_theta) + u_x * sin_theta, cos_theta + u_z ** 2 * (1 - cos_theta)]
                ])

                # 점 C를 회전
                rotated_C = np.dot(rotation_matrix, (C - B))  # B 기준으로 회전한 C
                #rotated_C = rotated_C / np.linalg.norm(rotated_C) * current_length  # 길이 고정
                #rotated_C = rotated_C * current_length / np.linalg.norm(original_BC_length) 
                new_C = B + rotated_C  # B를 중심으로 BC 길이 유지하며 새 점 계산
                positions.append(new_C)
                #각도 ABC 계산 벡터 내적
                BC = new_C - B
                cos_angle_ABC = np.dot(AB, BC) / (np.linalg.norm(AB) * np.linalg.norm(BC))
                angle_ABC = np.arccos(np.clip(cos_angle_ABC, -1.0, 1.0))  # 범위 클리핑

                # 각도가 이상이면 종료
                if np.degrees(angle_ABC) >= 170:
                    #print("ABC가 직선에 도달했습니다!")
                    break
        # 기존 vertices 배열과 positions 배열 합치기
        all_vertices = np.vstack((vertices, positions))

        # 새로 합친 vertices를 CSV로 저장
        output_file = "new_vertices.csv"
        pd.DataFrame(all_vertices).to_csv(output_file, header=["x","y","z"], index=False)

        print(f"새로운 vertices {output_file}로 저장")
        print(all_vertices.shape)

        csv_files = glob.glob(os.path.join(image_folder_path, "*.csv"))
        all_vertices = np.vstack((vertices, positions))

        #최종메시
        #파일 경로 설정
        vertices_file = 'vertices.csv'
        faces_file = 'faces.csv'

        # 데이터 로드
        vertices = pd.read_csv(vertices_file).to_numpy()
        # vertices = all_vertices
        faces = pd.read_csv(faces_file).to_numpy()
        # 점 클라우드 생성
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(positions))

        # 법선 추정
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=70.0, max_nn=90))
        point_cloud.orient_normals_consistent_tangent_plane(k=50)
        
        # 볼 피벗 메쉬 생성
        radii = [60, 80, 80]  # 다양한 반지름 시도
        mesh1 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            point_cloud,
            o3d.utility.DoubleVector(radii)
        )
        #  Open3D로 정점과 삼각형 추출
        vertices = np.asarray(mesh1.vertices)  # 정점 정보
        triangles = np.asarray(mesh1.triangles)  # 삼각형 면 정보

        vertices_df = pd.DataFrame(vertices, columns=["x", "y", "z"])
        vertices_df.to_csv("vertices_b.csv", index=False)
        print("vertices_b.csv 파일 저장 완료.")
        
        triangles_df = pd.DataFrame(triangles, columns=["v1", "v2", "v3"])  # 삼각형의 세 정점 인덱스
        triangles_df.to_csv("faces_b.csv", index=False)
        print("faces_b.csv 파일 저장 완료.")
        
        # vertices.csv에서 정점 데이터 읽기
        vertices = np.loadtxt('vertices_b.csv', delimiter=',', skiprows=1)
        # faces.csv에서 삼각형 면 데이터 읽기
        faces = np.loadtxt('faces_b.csv', delimiter=',', skiprows=1, dtype=int)

        # STL 메쉬 데이터 생성
        mesh_data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)
        for i, face in enumerate(faces):
            for j in range(3):
                mesh_data['vectors'][i][j] = vertices[face[j]]

        # STL 메쉬 객체 생성
        stl_mesh = mesh.Mesh(mesh_data)
        # STL 파일로 저장
        stl_mesh.save('output_b.stl')

        vertices_a = np.loadtxt('vertices.csv', delimiter=',', skiprows=1)
        faces_a = np.loadtxt('faces.csv', delimiter=',', skiprows=1, dtype=int)

        vertices_b = np.loadtxt('vertices_b.csv', delimiter=',', skiprows=1)
        faces_b = np.loadtxt('faces_b.csv', delimiter=',', skiprows=1, dtype=int)

        merged_vertices = np.vstack((vertices_a, vertices_b))

        #삼각형 데이터 합치기 (두 번째 메쉬의 인덱스에 첫 번째 메쉬의 정점 수를 더함)
        offset = len(vertices_a)  # 첫 번째 메쉬의 정점 개수
        merged_faces = np.vstack((faces_a, faces_b + offset))

        # 합쳐진 데이터로 새로운 메쉬 생성
        merged_mesh = o3d.geometry.TriangleMesh()
        merged_mesh.vertices = o3d.utility.Vector3dVector(merged_vertices)
        merged_mesh.triangles = o3d.utility.Vector3iVector(merged_faces)

        # 합쳐진 vertices.csv와 faces.csv로 저장
        pd.DataFrame(merged_vertices, columns=["x", "y", "z"]).to_csv('merged_vertices.csv', index=False)
        pd.DataFrame(merged_faces, columns=["v1", "v2", "v3"]).to_csv('merged_faces.csv', index=False)

        print("합친 메쉬 파일 저장")

        vertices = np.loadtxt('merged_vertices.csv', delimiter=',', skiprows=1)
        faces = np.loadtxt('merged_faces.csv', delimiter=',', skiprows=1, dtype=int)

        mesh_data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)

        for i, face in enumerate(faces):
            for j in range(3):
                mesh_data['vectors'][i][j] = vertices[face[j]]


        file_path = os.path.join(output_folder_path, f'output_b_merg{now_img_it+1}.stl')
        stl_mesh = mesh.Mesh(mesh_data)
        stl_mesh.save(file_path)

        print("STL 파일이 'output_b_merg.stl' 저장")


    finally:
        # if os.path.exists(vertices_path):
        #     os.remove(vertices_path)
        # if os.path.exists(faces_path):
        #     os.remove(faces_path)
        if os.path.exists(new_vertices_path):
            os.remove(new_vertices_path)
        if os.path.exists(input_image_path):
            os.rename(input_image_path,image_path)
        if os.path.exists(vertices_b_path):
            os.remove(vertices_b_path)
        if os.path.exists(faces_b_path):
            os.remove(faces_b_path)
        if os.path.exists(merged_vertices_path):
            os.remove(merged_vertices_path)
        if os.path.exists(merged_faces_path):
            os.remove(merged_faces_path)
        # if os.path.exists(output_b_stl_path):
        #     os.remove(output_b_stl_path)
        print("파이널리")

    print(f"사진{now_img_it+1}완료, 다음으로")