import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import matplotlib as mpl

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'NanumGothic'  # 나눔고딕 폰트 사용
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    print("나눔고딕 폰트가 설정되었습니다.")
except:
    print("나눔고딕 폰트를 찾을 수 없습니다. 폰트 설치가 필요할 수 있습니다.")
    try:
        # 대체 방법: 시스템에 있는 한글 지원 폰트 찾기
        font_list = mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        korean_fonts = [f for f in font_list if os.path.basename(f).startswith(('Malgun', 'malgun', 'Gulim', 'gulim', 'Batang', 'batang', 'Nanum', 'nanum', 'Dotum', 'dotum'))]
        
        if korean_fonts:
            plt.rcParams['font.family'] = mpl.font_manager.FontProperties(fname=korean_fonts[0]).get_name()
            plt.rcParams['axes.unicode_minus'] = False
            print(f"한글 지원 폰트를 찾았습니다: {os.path.basename(korean_fonts[0])}")
        else:
            print("한글 지원 폰트를 찾을 수 없습니다. 영문으로 표시됩니다.")
    except:
        print("폰트 설정 중 오류가 발생했습니다. 영문으로 표시됩니다.")

def load_point_cloud(file_path):
    """포인트 클라우드 파일을 로드합니다."""
    print(f"포인트 클라우드 파일을 로드합니다: {file_path}")
    point_cloud = o3d.io.read_point_cloud(file_path)
    print(f"로드된 포인트 수: {len(point_cloud.points)}")
    return point_cloud

def preprocess_point_cloud(pcd, voxel_size=0.02):
    """포인트 클라우드 전처리: 다운샘플링, 노이즈 제거"""
    print("포인트 클라우드 전처리 중...")
    
    # 다운샘플링
    print(f"다운샘플링 (복셀 크기: {voxel_size}m)...")
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # 노이즈 제거 (통계적 이상치 제거)
    print("노이즈 제거 중...")
    cleaned, _ = downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    return cleaned

def detect_main_plane(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
    """RANSAC 알고리즘을 사용하여 포인트 클라우드에서 가장 큰 평면(철골 면)을 감지합니다."""
    print("주요 철골 면 감지 중...")
    
    # 평면 감지 (RANSAC)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    # 평면에 해당하는 포인트들 추출
    plane_cloud = pcd.select_by_index(inliers)
    non_plane_cloud = pcd.select_by_index(inliers, invert=True)
    
    # 평면 모델 파라미터 (ax + by + cz + d = 0)
    a, b, c, d = plane_model
    
    print(f"감지된 평면 방정식: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"평면 포인트 수: {len(plane_cloud.points)}")
    print(f"물체(비평면) 포인트 수: {len(non_plane_cloud.points)}")
    
    # 평면 색상 지정 (회색)
    plane_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    
    return plane_model, plane_cloud, non_plane_cloud

def calculate_point_to_plane_distance(points, plane_model):
    """각 포인트에서 평면까지의 수직 거리를 계산합니다."""
    a, b, c, d = plane_model
    
    # 평면의 법선 벡터
    normal = np.array([a, b, c])
    normal_norm = np.linalg.norm(normal)
    
    # 각 포인트에서 평면까지의 수직 거리 계산
    # 거리 = |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
    distances = np.abs(np.dot(points, normal) + d) / normal_norm
    
    # 점이 평면의 어느 쪽에 있는지 결정 (양수: 법선 방향, 음수: 법선 반대 방향)
    signs = np.sign(np.dot(points, normal) + d)
    
    # 부호가 있는 거리 (방향 정보 포함)
    signed_distances = distances * signs
    
    return signed_distances

def color_points_by_distance(points, distances, cmap_name='jet', min_dist=None, max_dist=None):
    """거리에 따라 포인트에 색상을 할당합니다."""
    if min_dist is None:
        min_dist = np.min(distances)
    if max_dist is None:
        max_dist = np.max(distances)
    
    # 정규화된 거리 (0~1 사이)
    norm_distances = (distances - min_dist) / (max_dist - min_dist)
    
    # 컬러맵 적용
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(norm_distances)[:, :3]  # RGBA에서 RGB만 사용
    
    return colors

def create_distance_colored_point_cloud(points, colors):
    """거리에 따라 색상이 지정된 포인트 클라우드를 생성합니다."""
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    return colored_pcd

def visualize_result(plane_cloud, distance_colored_cloud, normal_vector, plane_model, max_distance=0.2):
    """결과를 시각화합니다."""
    print("결과 시각화 중...")
    
    # 좌표축 생성
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    
    # 법선 벡터 시각화 (평면의 중심에서 법선 방향으로 화살표)
    plane_center = np.mean(np.asarray(plane_cloud.points), axis=0)
    
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal) * 0.3  # 0.3m 길이로 스케일링
    
    # 법선 벡터를 나타내는 라인 생성
    line_points = np.array([plane_center, plane_center + normal])
    line_indices = np.array([[0, 1]])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_indices)
    line_set.paint_uniform_color([1, 0, 0])  # 빨간색
    
    # 시각화
    o3d.visualization.draw_geometries([
        plane_cloud,
        distance_colored_cloud,
        coordinate_frame,
        line_set
    ], window_name="철골 면으로부터의 거리 시각화")

def create_color_bar(distances, cmap_name='jet', min_dist=None, max_dist=None, output_dir="results"):
    """거리에 대한 컬러바를 생성하고 이미지로 저장합니다."""
    if min_dist is None:
        min_dist = np.min(distances)
    if max_dist is None:
        max_dist = np.max(distances)
    
    # 컬러바 생성
    fig, ax = plt.subplots(figsize=(8, 1))
    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(min_dist, max_dist)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                     cax=ax, orientation='horizontal')
    cb.set_label('철골 면으로부터의 거리 (m)')
    
    # 이미지 저장
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "distance_colorbar.png")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"컬러바 이미지가 {output_path}에 저장되었습니다.")
    plt.close()

def analyze_distance_statistics(distances):
    """거리 통계를 분석합니다."""
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    std_dist = np.std(distances)
    
    print("\n=== 거리 통계 분석 ===")
    print(f"평균 거리: {mean_dist:.4f}m")
    print(f"중앙값 거리: {median_dist:.4f}m")
    print(f"최소 거리: {min_dist:.4f}m")
    print(f"최대 거리: {max_dist:.4f}m")
    print(f"표준 편차: {std_dist:.4f}m")
    
    # 히스토그램 생성
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, color='skyblue', edgecolor='black')
    plt.title('철골 면으로부터의 거리 분포')
    plt.xlabel('거리 (m)')
    plt.ylabel('포인트 개수')
    plt.grid(alpha=0.3)
    
    # 이미지 저장
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "distance_histogram.png")
    plt.savefig(output_path)
    print(f"히스토그램이 {output_path}에 저장되었습니다.")
    plt.close()
    
    return {
        'mean': mean_dist,
        'median': median_dist,
        'min': min_dist,
        'max': max_dist,
        'std': std_dist
    }

def save_distance_colored_point_cloud(pcd, output_dir="results", filename="distance_colored_objects.ply"):
    """거리에 따라 색상이 지정된 포인트 클라우드를 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"거리 색상화된 포인트 클라우드가 {output_path}에 저장되었습니다.")

def main():
    # 포인트 클라우드 파일 경로
    input_path = "results/3_pointcloud_filtered_2m.ply"
    if not os.path.exists(input_path):
        input_path = "3_pointcloud_filtered_2.0m.ply"
        if not os.path.exists(input_path):
            print("포인트 클라우드 파일을 찾을 수 없습니다.")
            return
    
    # 출력 디렉토리
    output_dir = "results/distance_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 포인트 클라우드 로드
    pcd = load_point_cloud(input_path)
    
    # 2. 포인트 클라우드 전처리
    preprocessed_pcd = preprocess_point_cloud(pcd, voxel_size=0.01)  # 더 정밀한 분석을 위해 복셀 크기 감소
    
    # 3. 주요 철골 면 감지
    distance_threshold = float(input("평면 감지 임계값을 입력하세요 (기본값: 0.01m): ") or "0.01")
    plane_model, plane_cloud, non_plane_cloud = detect_main_plane(
        preprocessed_pcd, 
        distance_threshold=distance_threshold
    )
    
    # 4. 물체(비평면) 포인트에서 철골 면까지의 거리 계산
    object_points = np.asarray(non_plane_cloud.points)
    distances = calculate_point_to_plane_distance(object_points, plane_model)
    
    # 5. 거리에 따라 포인트 색상 지정
    # 사용자 지정 최소/최대 거리 (기본값: 데이터의 min/max)
    use_custom_range = input("거리 색상 범위를 지정하시겠습니까? (y/n): ").lower() == 'y'
    
    if use_custom_range:
        min_dist = float(input("최소 거리(m)를 입력하세요: ") or "0")
        max_dist = float(input("최대 거리(m)를 입력하세요: ") or str(np.max(distances)))
    else:
        # 기본 범위: 최소값과 최대값을 데이터에서 결정
        min_dist = np.min(distances)
        max_dist = np.max(distances)
    
    # 6. 거리에 따른 색상 매핑
    colors = color_points_by_distance(object_points, distances, min_dist=min_dist, max_dist=max_dist)
    distance_colored_cloud = create_distance_colored_point_cloud(object_points, colors)
    
    # 7. 거리 통계 분석
    stats = analyze_distance_statistics(distances)
    
    # 8. 컬러바 생성
    create_color_bar(distances, min_dist=min_dist, max_dist=max_dist, output_dir=output_dir)
    
    # 9. 결과 저장
    save_distance_colored_point_cloud(distance_colored_cloud, output_dir=output_dir)
    
    # 10. 결과 시각화
    a, b, c, d = plane_model
    normal_vector = np.array([a, b, c])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    visualize_result(plane_cloud, distance_colored_cloud, normal_vector, plane_model, max_distance=max_dist)
    
    print("\n분석이 완료되었습니다.")
    print(f"- 철골 면 포인트 수: {len(plane_cloud.points)}")
    print(f"- 물체(비평면) 포인트 수: {len(non_plane_cloud.points)}")
    print(f"- 거리 범위: {min_dist:.4f}m ~ {max_dist:.4f}m")
    print(f"- 평균 거리: {stats['mean']:.4f}m")
    print(f"결과 파일이 {output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main() 