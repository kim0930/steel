import open3d as o3d
import numpy as np
import os

# 포인트 클라우드 파일 경로
input_path = "results/3_pointcloud.ply"
if not os.path.exists(input_path):
    input_path = "3_pointcloud.ply"
    if not os.path.exists(input_path):
        print("3_pointcloud.ply 파일을 찾을 수 없습니다.")
        exit(1)

# 결과 파일 경로
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "3_pointcloud_filtered_1m.ply")

# 포인트 클라우드 로드
print(f"포인트 클라우드 파일을 로드합니다: {input_path}")
point_cloud = o3d.io.read_point_cloud(input_path)
print(f"로드된 포인트 수: {len(point_cloud.points)}")

# 원점 좌표 설정 - 원점(0,0,0) 사용
origin = [0, 0, 0]
max_distance = 1.0  # 1m 설정

# 포인트 클라우드 필터링
points = np.asarray(point_cloud.points)
colors = np.asarray(point_cloud.colors)

# 각 점과 원점 사이의 거리 계산
distances = np.sqrt(np.sum((points - origin) ** 2, axis=1))

# max_distance 이내의 점들만 선택
mask = distances <= max_distance
filtered_points = points[mask]
filtered_colors = colors[mask]

# 새로운 포인트 클라우드 생성
filtered_cloud = o3d.geometry.PointCloud()
filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)
filtered_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

print(f"원래 포인트 수: {len(points)}")
print(f"필터링된 포인트 수: {len(filtered_points)}")
print(f"필터링 비율: {len(filtered_points) / len(points) * 100:.2f}%")

# 결과 저장
o3d.io.write_point_cloud(output_path, filtered_cloud)
print(f"필터링된 포인트 클라우드가 {output_path}에 저장되었습니다.")

# 시각화
print("필터링된 포인트 클라우드를 시각화합니다. 창을 닫으면 종료됩니다.")

# 좌표축 생성
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=origin)

# 원점 표시 구체
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
sphere.paint_uniform_color([1, 0, 0])  # 빨간색

# 1m 거리 표시 와이어프레임
distance_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max_distance)
distance_sphere.compute_vertex_normals()
wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(distance_sphere)
wireframe.paint_uniform_color([0, 0.8, 0])  # 초록색

# 시각화
o3d.visualization.draw_geometries([filtered_cloud, coordinate_frame, sphere, wireframe], 
                               window_name="1m 이내 필터링된 포인트 클라우드") 