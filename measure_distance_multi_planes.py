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

def detect_multiple_planes(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000, 
                          min_plane_size=100, max_planes=5):
    """RANSAC 알고리즘을 사용하여 포인트 클라우드에서 여러 평면(철골 면)을 감지합니다."""
    print("철골 면 감지 중...")
    
    planes = []  # 감지된 평면 클라우드 리스트
    plane_models = []  # 감지된 평면 방정식 리스트
    plane_colors = []  # 각 평면의 색상
    remaining_cloud = pcd  # 남은 포인트 클라우드
    
    for i in range(max_planes):
        if len(remaining_cloud.points) < min_plane_size:
            print(f"남은 포인트가 너무 적습니다 ({len(remaining_cloud.points)} < {min_plane_size}). 평면 감지를 중단합니다.")
            break
            
        print(f"평면 {i+1} 감지 중... (남은 포인트: {len(remaining_cloud.points)})")
        
        # 평면 감지 (RANSAC)
        plane_model, inliers = remaining_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        if len(inliers) < min_plane_size:
            print(f"감지된 평면의 포인트가 너무 적습니다 ({len(inliers)} < {min_plane_size}). 평면 감지를 중단합니다.")
            break
            
        # 평면에 해당하는 포인트들 추출
        plane_cloud = remaining_cloud.select_by_index(inliers)
        
        # 평면 모델 파라미터 (ax + by + cz + d = 0)
        a, b, c, d = plane_model
        
        # 평면 색상 (고유한 색상 할당)
        color = [random.random(), random.random(), random.random()]
        plane_cloud.paint_uniform_color(color)
        
        # 평면 정보 저장
        planes.append(plane_cloud)
        plane_models.append(plane_model)
        plane_colors.append(color)
        
        print(f"평면 {i+1} 감지 완료: {len(inliers)}개 포인트")
        print(f"평면 방정식: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        
        # 다음 평면 감지를 위해 남은 포인트 업데이트
        remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)
    
    # 물체(남은 포인트) 클라우드
    object_cloud = remaining_cloud
    
    print(f"총 {len(planes)}개의 평면을 감지했습니다.")
    print(f"남은 물체 포인트 수: {len(object_cloud.points)}")
    
    return planes, plane_models, plane_colors, object_cloud

def calculate_point_to_plane_distance(points, plane_model):
    """각 포인트에서 평면까지의 수직 거리를 계산합니다."""
    a, b, c, d = plane_model
    
    # 평면의 법선 벡터
    normal = np.array([a, b, c])
    normal_norm = np.linalg.norm(normal)
    
    # 각 포인트에서 평면까지의 수직 거리 계산
    distances = np.abs(np.dot(points, normal) + d) / normal_norm
    
    # 점이 평면의 어느 쪽에 있는지 결정 (양수: 법선 방향, 음수: 법선 반대 방향)
    signs = np.sign(np.dot(points, normal) + d)
    
    # 부호가 있는 거리 (방향 정보 포함)
    signed_distances = distances * signs
    
    return signed_distances

def find_nearest_plane_distances(points, plane_models):
    """각 포인트에 대해 가장 가까운 평면까지의 거리를 계산합니다."""
    # 모든 평면에 대한 거리 계산
    all_distances = []
    for plane_model in plane_models:
        distances = calculate_point_to_plane_distance(points, plane_model)
        all_distances.append(distances)
    
    # 행렬 변환 (평면 수 x 포인트 수)
    distance_matrix = np.vstack(all_distances)
    
    # 각 포인트에 대해 가장 가까운 평면 찾기
    min_distances = np.min(np.abs(distance_matrix), axis=0)
    nearest_plane_indices = np.argmin(np.abs(distance_matrix), axis=0)
    
    # 원래 부호 복원 (가장 가까운 평면에 대한 부호)
    signed_min_distances = np.array([
        distance_matrix[plane_idx, point_idx] 
        for point_idx, plane_idx in enumerate(nearest_plane_indices)
    ])
    
    return min_distances, signed_min_distances, nearest_plane_indices

def color_points_by_distance(points, distances, cmap_name='jet', min_dist=None, max_dist=None):
    """거리에 따라 포인트에 색상을 할당합니다."""
    if min_dist is None:
        min_dist = np.min(distances)
    if max_dist is None:
        max_dist = np.max(distances)
    
    # 정규화된 거리 (0~1 사이)
    norm_distances = np.clip((distances - min_dist) / (max_dist - min_dist), 0, 1)
    
    # 컬러맵 적용
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(norm_distances)[:, :3]  # RGBA에서 RGB만 사용
    
    return colors

def color_points_by_nearest_plane(points, nearest_plane_indices, plane_colors):
    """가장 가까운 평면에 따라 포인트에 색상을 할당합니다."""
    colors = np.zeros((len(points), 3))
    
    for i, plane_idx in enumerate(nearest_plane_indices):
        colors[i] = plane_colors[plane_idx]
    
    return colors

def create_colored_point_cloud(points, colors):
    """색상이 지정된 포인트 클라우드를 생성합니다."""
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    return colored_pcd

def visualize_planes_and_distances(planes, object_cloud, plane_models, output_dir="results"):
    """감지된 평면과 물체의 거리를 시각화합니다."""
    print("결과 시각화 중...")
    
    # 좌표축 생성
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    
    # 모든 평면과 물체, 좌표축을 포함하는 리스트 생성
    geometries = planes + [object_cloud, coordinate_frame]
    
    # 시각화
    o3d.visualization.draw_geometries(geometries, window_name="철골 면과 물체")
    
    # 결과 저장 (전체 장면)
    combined_cloud = o3d.geometry.PointCloud()
    for plane in planes:
        combined_cloud += plane
    combined_cloud += object_cloud
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "planes_and_objects.ply")
    o3d.io.write_point_cloud(output_path, combined_cloud)
    print(f"전체 장면이 {output_path}에 저장되었습니다.")

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

def analyze_distance_statistics(distances, plane_indices, num_planes):
    """거리 통계를 분석합니다."""
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    std_dist = np.std(distances)
    
    print("\n=== 전체 거리 통계 분석 ===")
    print(f"평균 거리: {mean_dist:.4f}m")
    print(f"중앙값 거리: {median_dist:.4f}m")
    print(f"최소 거리: {min_dist:.4f}m")
    print(f"최대 거리: {max_dist:.4f}m")
    print(f"표준 편차: {std_dist:.4f}m")
    
    # 각 평면별 통계
    print("\n=== 평면별 거리 통계 ===")
    for plane_idx in range(num_planes):
        plane_mask = (plane_indices == plane_idx)
        if np.sum(plane_mask) > 0:
            plane_distances = distances[plane_mask]
            print(f"\n평면 {plane_idx+1}:")
            print(f"  포인트 수: {len(plane_distances)}")
            print(f"  평균 거리: {np.mean(plane_distances):.4f}m")
            print(f"  중앙값 거리: {np.median(plane_distances):.4f}m")
            print(f"  최소 거리: {np.min(plane_distances):.4f}m")
            print(f"  최대 거리: {np.max(plane_distances):.4f}m")
    
    # 히스토그램 생성
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, color='skyblue', edgecolor='black')
    plt.title('철골 면으로부터의 거리 분포')
    plt.xlabel('거리 (m)')
    plt.ylabel('포인트 개수')
    plt.grid(alpha=0.3)
    
    # 이미지 저장
    output_dir = "results/multi_plane_analysis"
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

def save_colored_point_clouds(distance_colored_cloud, plane_colored_cloud, output_dir="results/multi_plane_analysis"):
    """색상화된 포인트 클라우드를 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 거리 기반 색상 포인트 클라우드 저장
    distance_output_path = os.path.join(output_dir, "distance_colored_objects.ply")
    o3d.io.write_point_cloud(distance_output_path, distance_colored_cloud)
    print(f"거리 색상화된 포인트 클라우드가 {distance_output_path}에 저장되었습니다.")
    
    # 평면 기반 색상 포인트 클라우드 저장
    plane_output_path = os.path.join(output_dir, "plane_colored_objects.ply")
    o3d.io.write_point_cloud(plane_output_path, plane_colored_cloud)
    print(f"평면 색상화된 포인트 클라우드가 {plane_output_path}에 저장되었습니다.")

def main():
    # 포인트 클라우드 파일 경로
    input_path = "results/3_pointcloud_filtered_2m.ply"
    if not os.path.exists(input_path):
        input_path = "3_pointcloud_filtered_2.0m.ply"
        if not os.path.exists(input_path):
            print("포인트 클라우드 파일을 찾을 수 없습니다.")
            return
    
    # 출력 디렉토리
    output_dir = "results/multi_plane_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 포인트 클라우드 로드
    pcd = load_point_cloud(input_path)
    
    # 2. 포인트 클라우드 전처리
    preprocessed_pcd = preprocess_point_cloud(pcd, voxel_size=0.01)  # 더 정밀한 분석을 위해 복셀 크기 감소
    
    # 3. 여러 철골 면 감지
    distance_threshold = float(input("평면 감지 임계값을 입력하세요 (기본값: 0.01m): ") or "0.01")
    max_planes = int(input("최대 감지할 평면 수를 입력하세요 (기본값: 5): ") or "5")
    min_plane_size = int(input("최소 평면 크기를 입력하세요 (기본값: 100 포인트): ") or "100")
    
    planes, plane_models, plane_colors, object_cloud = detect_multiple_planes(
        preprocessed_pcd, 
        distance_threshold=distance_threshold,
        min_plane_size=min_plane_size,
        max_planes=max_planes
    )
    
    if not planes:
        print("평면을 감지하지 못했습니다. 프로그램을 종료합니다.")
        return
    
    # 4. 물체(비평면) 포인트에서 가장 가까운 철골 면까지의 거리 계산
    object_points = np.asarray(object_cloud.points)
    min_distances, signed_distances, nearest_plane_indices = find_nearest_plane_distances(
        object_points, plane_models
    )
    
    # 5. 거리에 따라 포인트 색상 지정
    # 사용자 지정 최소/최대 거리 (기본값: 데이터의 min/max)
    use_custom_range = input("거리 색상 범위를 지정하시겠습니까? (y/n): ").lower() == 'y'
    
    if use_custom_range:
        min_dist = float(input("최소 거리(m)를 입력하세요: ") or "0")
        max_dist = float(input("최대 거리(m)를 입력하세요: ") or str(np.max(min_distances)))
    else:
        # 기본 범위: 최소값과 최대값을 데이터에서 결정
        min_dist = np.min(min_distances)
        max_dist = np.max(min_distances)
    
    # 6. 거리에 따른 색상 매핑
    distance_colors = color_points_by_distance(object_points, min_distances, min_dist=min_dist, max_dist=max_dist)
    distance_colored_cloud = create_colored_point_cloud(object_points, distance_colors)
    
    # 7. 가장 가까운 평면에 따른 색상 매핑
    plane_based_colors = color_points_by_nearest_plane(object_points, nearest_plane_indices, plane_colors)
    plane_colored_cloud = create_colored_point_cloud(object_points, plane_based_colors)
    
    # 8. 거리 통계 분석
    stats = analyze_distance_statistics(min_distances, nearest_plane_indices, len(planes))
    
    # 9. 컬러바 생성
    create_color_bar(min_distances, min_dist=min_dist, max_dist=max_dist, output_dir=output_dir)
    
    # 10. 결과 저장
    save_colored_point_clouds(distance_colored_cloud, plane_colored_cloud, output_dir=output_dir)
    
    # 11. 결과 시각화
    # 원래 감지된 평면과 물체
    visualize_planes_and_distances(planes, object_cloud, plane_models, output_dir=output_dir)
    
    # 거리에 따라 색상화된 물체
    print("\n거리에 따라 색상화된 물체 시각화 중...")
    combined_geometries = planes + [distance_colored_cloud, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]
    o3d.visualization.draw_geometries(combined_geometries, window_name="철골 면과 거리 색상화된 물체")
    
    # 가장 가까운 평면에 따라 색상화된 물체
    print("\n가장 가까운 평면에 따라 색상화된 물체 시각화 중...")
    plane_colored_geometries = planes + [plane_colored_cloud, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]
    o3d.visualization.draw_geometries(plane_colored_geometries, window_name="철골 면과 평면 색상화된 물체")
    
    print("\n분석이 완료되었습니다.")
    print(f"- 감지된 철골 면 수: {len(planes)}")
    print(f"- 물체(비평면) 포인트 수: {len(object_cloud.points)}")
    print(f"- 거리 범위: {min_dist:.4f}m ~ {max_dist:.4f}m")
    print(f"- 평균 거리: {stats['mean']:.4f}m")
    print(f"결과 파일이 {output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main() 