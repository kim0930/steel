import open3d as o3d
import numpy as np
import os

def load_point_cloud(file_path):
    """포인트 클라우드 파일을 로드합니다."""
    print(f"포인트 클라우드 파일을 로드합니다: {file_path}")
    point_cloud = o3d.io.read_point_cloud(file_path)
    print(f"로드된 포인트 수: {len(point_cloud.points)}")
    return point_cloud

def filter_points_by_distance(point_cloud, max_distance=1.0, origin=[0, 0, 0]):
    """원점으로부터 지정된 거리 이내의 점들만 필터링합니다."""
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    
    # 각 점과 원점 사이의 거리 계산
    distances = np.sqrt(np.sum((points - origin) ** 2, axis=1))
    
    # max_distance 이내의 점들만 선택
    mask = distances <= max_distance
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    
    # 새로운 포인트 클라우드 생성
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    print(f"원래 포인트 수: {len(points)}")
    print(f"필터링된 포인트 수: {len(filtered_points)}")
    print(f"필터링 비율: {len(filtered_points) / len(points) * 100:.2f}%")
    
    return filtered_point_cloud

def pick_points(pcd):
    """사용자가 포인트 클라우드에서 점을 선택하도록 합니다."""
    print("기준점을 선택하세요. [Shift+클릭]으로 점을 선택하고, [Q]를 눌러 종료하세요.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # 사용자가 점을 선택할 때까지 대기
    vis.destroy_window()
    
    # 선택된 점 인덱스를 가져옴
    picked_indices = vis.get_picked_points()
    
    if not picked_indices:
        print("점이 선택되지 않았습니다. 원점(0,0,0)을 사용합니다.")
        return [0, 0, 0]
    
    # 첫 번째 선택된 점의 좌표를 반환
    points = np.asarray(pcd.points)
    picked_point = points[picked_indices[0]]
    print(f"선택한 점의 좌표: {picked_point}")
    
    return picked_point

def visualize_point_cloud(point_cloud, origin=[0, 0, 0], max_distance=1.0, window_name="필터링된 포인트 클라우드"):
    """포인트 클라우드를 시각화합니다."""
    # 좌표축 생성 (크기 1m로 설정하여 1m 기준 시각화)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=origin)
    
    # 기준점을 표시하는 구체 생성 (반지름 0.02m)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    sphere.translate(origin)
    sphere.paint_uniform_color([1, 0, 0])  # 빨간색으로 설정
    
    # 거리 기준의 와이어프레임 구체 생성 (필터링 거리 시각화)
    distance_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max_distance)
    distance_sphere.translate(origin)
    distance_sphere.compute_vertex_normals()
    distance_sphere.paint_uniform_color([0, 1, 0])  # 초록색으로 설정
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(distance_sphere)
    wireframe.paint_uniform_color([0, 0.8, 0])  # 조금 더 어두운 초록색
    
    # 시각화
    print("포인트 클라우드를 시각화합니다. 창을 닫으면 계속 진행됩니다.")
    o3d.visualization.draw_geometries([point_cloud, coordinate_frame, sphere, wireframe], window_name=window_name)

def save_point_cloud(point_cloud, output_path):
    """포인트 클라우드를 파일로 저장합니다."""
    o3d.io.write_point_cloud(output_path, point_cloud)
    print(f"필터링된 포인트 클라우드가 {output_path}에 저장되었습니다.")

def main():
    # 포인트 클라우드 파일 경로
    pointcloud_path = os.path.join("results", "3_pointcloud.ply")
    
    # 결과 파일 경로
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 포인트 클라우드 로드
    try:
        point_cloud = load_point_cloud(pointcloud_path)
    except Exception as e:
        print(f"첫 번째 경로에서 오류 발생: {str(e)}")
        
        # 파일이 results 폴더에 없는 경우, 루트 디렉토리에서 찾기 시도
        try:
            pointcloud_path = "3_pointcloud.ply"
            print(f"다음 경로에서 다시 시도합니다: {pointcloud_path}")
            point_cloud = load_point_cloud(pointcloud_path)
        except Exception as e2:
            print(f"두 번째 시도에서도 오류 발생: {str(e2)}")
            print("3_pointcloud.ply 파일을 찾을 수 없습니다.")
            return
    
    # 전체 포인트 클라우드 시각화
    print("전체 포인트 클라우드를 시각화합니다.")
    visualize_point_cloud(point_cloud, origin=[0, 0, 0], max_distance=1.0, window_name="원본 포인트 클라우드")
    
    # 사용자에게 기준점 선택 옵션 제공
    use_custom_origin = input("기준점을 직접 선택하시겠습니까? (y/n): ").lower() == 'y'
    
    if use_custom_origin:
        # 사용자가 포인트 클라우드에서 기준점 선택
        origin = pick_points(point_cloud)
    else:
        # 기본 원점 사용
        origin = [0, 0, 0]
        print("기본 원점(0,0,0)을 사용합니다.")
    
    # 거리 기준 입력 받기
    try:
        max_distance = float(input(f"거리 필터링 기준(미터)을 입력하세요 (기본값 1.0m): ") or "1.0")
    except ValueError:
        max_distance = 1.0
        print("잘못된 입력입니다. 기본값 1.0m를 사용합니다.")
    
    # 결과 파일 경로 설정
    output_path = os.path.join(output_dir, f"3_pointcloud_filtered_{max_distance}m.ply")
    
    # 지정된 거리 이내의 점들만 필터링
    filtered_point_cloud = filter_points_by_distance(point_cloud, max_distance=max_distance, origin=origin)
    
    # 필터링된 포인트 클라우드 저장
    save_point_cloud(filtered_point_cloud, output_path)
    
    # 필터링된 포인트 클라우드 시각화
    visualize_point_cloud(filtered_point_cloud, origin=origin, max_distance=max_distance, 
                        window_name=f"{max_distance}m 이내 필터링된 포인트 클라우드")

if __name__ == "__main__":
    main() 