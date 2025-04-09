import open3d as o3d
import numpy as np
import os
import random
import matplotlib.pyplot as plt

def load_point_cloud(file_path):
    """포인트 클라우드 파일을 로드합니다."""
    print(f"포인트 클라우드 파일을 로드합니다: {file_path}")
    point_cloud = o3d.io.read_point_cloud(file_path)
    print(f"로드된 포인트 수: {len(point_cloud.points)}")
    return point_cloud

def preprocess_point_cloud(pcd, voxel_size=0.02):
    """포인트 클라우드 전처리: 다운샘플링, 노이즈 제거, 법선 벡터 계산"""
    print("포인트 클라우드 전처리 중...")
    
    # 다운샘플링
    print(f"다운샘플링 (복셀 크기: {voxel_size}m)...")
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # 노이즈 제거 (통계적 이상치 제거)
    print("노이즈 제거 중...")
    cleaned, _ = downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 법선 벡터 계산
    print("법선 벡터 계산 중...")
    cleaned.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    return cleaned

def detect_planes(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000, min_plane_size=100):
    """RANSAC 알고리즘을 사용하여 포인트 클라우드에서 평면을 감지합니다."""
    print("평면 감지 중...")
    planes = []
    plane_colors = []
    rest = pcd
    
    # 각 평면에 대한 랜덤 색상 생성 함수
    def get_random_color():
        return [random.random(), random.random(), random.random()]
    
    # 일정 크기 이상의 평면이 더 이상 감지되지 않을 때까지 반복
    segment_models = []
    segments = []
    max_planes = 10  # 최대 평면 수 제한
    remaining_points = len(rest.points)
    
    for i in range(max_planes):
        if remaining_points < min_plane_size:
            break
            
        print(f"평면 {i+1} 감지 중... (남은 포인트: {remaining_points})")
        
        # 평면 감지 (RANSAC)
        plane_model, inliers = rest.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        if len(inliers) < min_plane_size:
            print(f"더 이상 충분한 크기의 평면이 없습니다. (마지막 크기: {len(inliers)})")
            break
            
        # 평면에 해당하는 포인트들 추출
        plane_cloud = rest.select_by_index(inliers)
        remaining_cloud = rest.select_by_index(inliers, invert=True)
        
        # 평면 색상 지정
        color = get_random_color()
        plane_cloud.paint_uniform_color(color)
        
        # 평면 정보 저장
        segment_models.append(plane_model)
        segments.append(plane_cloud)
        planes.append(plane_cloud)
        plane_colors.append(color)
        
        # 다음 평면 감지를 위해 남은 포인트 업데이트
        rest = remaining_cloud
        remaining_points = len(rest.points)
        
        print(f"평면 {i+1} 감지 완료: {len(inliers)}개 포인트, 평면 방정식: {plane_model}")
    
    print(f"총 {len(planes)}개의 평면이 감지되었습니다.")
    
    # 남은 포인트들은 회색으로 표시
    if len(rest.points) > 0:
        rest.paint_uniform_color([0.5, 0.5, 0.5])  # 회색
        planes.append(rest)
        plane_colors.append([0.5, 0.5, 0.5])
    
    return planes, plane_colors, segment_models

def visualize_planes(planes, colors, window_name="감지된 평면"):
    """감지된 평면을 시각화합니다."""
    print("감지된 평면을 시각화합니다. 창을 닫으면 계속 진행됩니다.")
    
    # 좌표축 생성
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    # 모든 평면과 좌표축을 포함하는 리스트 생성
    geometries = planes + [coordinate_frame]
    
    # 시각화
    o3d.visualization.draw_geometries(geometries, window_name=window_name)

def analyze_planes(planes, models):
    """감지된 평면들을 분석합니다."""
    print("\n=== 감지된 평면 분석 ===")
    
    for i, (plane, model) in enumerate(zip(planes, models)):
        # 평면의 법선 벡터
        normal = model[:3]
        
        # 평면의 면적 추정 (바운딩 박스 사용)
        points = np.asarray(plane.points)
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        dimensions = max_bound - min_bound
        
        # 평면이 어느 축에 가장 평행한지 확인
        abs_normal = np.abs(normal)
        dominant_axis = np.argmax(abs_normal)
        axis_names = ['X', 'Y', 'Z']
        
        # X, Y, Z 축과의 각도 계산
        angle_x = np.arccos(np.abs(normal[0]) / np.linalg.norm(normal)) * 180 / np.pi
        angle_y = np.arccos(np.abs(normal[1]) / np.linalg.norm(normal)) * 180 / np.pi
        angle_z = np.arccos(np.abs(normal[2]) / np.linalg.norm(normal)) * 180 / np.pi
        
        print(f"\n평면 {i+1}:")
        print(f"  포인트 수: {len(plane.points)}")
        print(f"  법선 벡터: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
        print(f"  평면 방정식: {normal[0]:.4f}x + {normal[1]:.4f}y + {normal[2]:.4f}z + {model[3]:.4f} = 0")
        print(f"  주요 축: {axis_names[dominant_axis]}")
        print(f"  축과의 각도: X: {angle_x:.2f}°, Y: {angle_y:.2f}°, Z: {angle_z:.2f}°")
        print(f"  대략적인 크기: {dimensions[0]:.2f}m x {dimensions[1]:.2f}m x {dimensions[2]:.2f}m")

def save_plane_clouds(planes, output_dir="results/planes"):
    """감지된 각 평면을 개별 파일로 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, plane in enumerate(planes):
        output_path = os.path.join(output_dir, f"plane_{i+1}.ply")
        o3d.io.write_point_cloud(output_path, plane)
        print(f"평면 {i+1}이 {output_path}에 저장되었습니다.")

def auto_adjust_threshold(pcd, initial_threshold=0.02, step=0.005, min_threshold=0.005, max_threshold=0.1, 
                        target_planes=5, max_iterations=10):
    """
    평면 감지를 위한 임계값을 자동으로 조정합니다.
    적절한 수의 평면이 감지될 때까지 임계값을 조정합니다.
    """
    print("평면 감지 임계값 자동 조정 중...")
    
    threshold = initial_threshold
    best_threshold = initial_threshold
    best_count = 0
    iterations = 0
    
    while iterations < max_iterations:
        # 현재 임계값으로 평면 감지
        rest = pcd
        plane_count = 0
        
        for i in range(target_planes * 2):  # 목표의 2배까지 시도
            if len(rest.points) < 100:  # 최소 포인트 수
                break
                
            # 평면 감지 시도
            plane_model, inliers = rest.segment_plane(
                distance_threshold=threshold,
                ransac_n=3,
                num_iterations=1000
            )
            
            if len(inliers) < 100:  # 작은 평면은 무시
                break
                
            # 평면으로 인식된 포인트 제외
            plane_count += 1
            rest = rest.select_by_index(inliers, invert=True)
        
        print(f"  임계값 {threshold:.4f}m에서 {plane_count}개 평면 감지됨")
        
        # 목표 평면 수와의 차이 계산
        diff = abs(plane_count - target_planes)
        
        # 현재까지의 가장 좋은 결과 업데이트
        if best_count == 0 or diff < abs(best_count - target_planes):
            best_threshold = threshold
            best_count = plane_count
        
        # 목표에 충분히 가까우면 중단
        if diff <= 1:
            print(f"적절한 임계값을 찾았습니다: {threshold:.4f}m (감지된 평면 수: {plane_count})")
            return threshold
        
        # 임계값 조정
        if plane_count < target_planes:
            # 평면이 적게 감지되면 임계값을 증가 (더 느슨하게)
            threshold = min(threshold + step, max_threshold)
        else:
            # 평면이 많이 감지되면 임계값을 감소 (더 엄격하게)
            threshold = max(threshold - step, min_threshold)
        
        iterations += 1
    
    print(f"자동 조정 종료. 최적 임계값: {best_threshold:.4f}m (감지된 평면 수: {best_count})")
    return best_threshold

def visualize_result_comparison(original_pcd, planes, output_dir="results"):
    """원본 포인트 클라우드와 감지된 평면을 비교 시각화합니다."""
    print("결과 비교 시각화 중...")
    
    # 감지된 모든 평면을 결합
    combined = o3d.geometry.PointCloud()
    for plane in planes:
        combined += plane
    
    # 원본 포인트 클라우드와 감지된 평면을 나란히 시각화
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name="원본 포인트 클라우드", width=800, height=600, left=0, top=0)
    vis1.add_geometry(original_pcd)
    vis1.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="감지된 평면", width=800, height=600, left=800, top=0)
    
    # 감지된 모든 평면 추가
    for plane in planes:
        vis2.add_geometry(plane)
    
    vis2.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    
    # 렌더링 옵션 설정
    vis1_opt = vis1.get_render_option()
    vis1_opt.point_size = 2.0
    
    vis2_opt = vis2.get_render_option()
    vis2_opt.point_size = 2.0
    
    # 카메라 제어
    vis1_ctrl = vis1.get_view_control()
    vis2_ctrl = vis2.get_view_control()
    
    # 카메라 동기화 함수
    def update_cameras():
        # vis1의 카메라 설정을 vis2에 복사
        param = vis1_ctrl.convert_to_pinhole_camera_parameters()
        vis2_ctrl.convert_from_pinhole_camera_parameters(param)
    
    # 화면 갱신 및 이벤트 처리
    while True:
        vis1.update_geometry(original_pcd)
        vis2.update_geometry(planes[0])  # 대표적인 하나의 평면만 업데이트
        
        update_cameras()
        
        if not vis1.poll_events() or not vis2.poll_events():
            break
            
        vis1.update_renderer()
        vis2.update_renderer()
    
    # 창 닫기
    vis1.destroy_window()
    vis2.destroy_window()
    
    # 결과 이미지 저장
    output_path = os.path.join(output_dir, "plane_detection_result.png")
    print(f"결과 이미지가 {output_path}에 저장되었습니다.")

def classify_steel_structure_planes(planes, models):
    """감지된 평면들을 철골구조물의 부분으로 분류합니다."""
    print("\n=== 철골구조물 평면 분류 ===")
    
    # 분류 결과를 저장할 딕셔너리
    classification = {
        'horizontal': [],  # 수평면 (바닥/천장)
        'vertical_main': [],  # 주요 수직면 (기둥)
        'vertical_secondary': [],  # 보조 수직면
        'diagonal': [],  # 사선면 (브레이싱)
        'unknown': []  # 분류되지 않은 면
    }
    
    # 각 평면의 분류 레이블
    plane_labels = []
    
    # 평면들의 법선 벡터를 분석하여 분류
    for i, (plane, model) in enumerate(zip(planes, models)):
        if i >= len(models):  # 마지막 '기타' 부분 제외
            plane_labels.append('other')
            continue
            
        # 평면의 법선 벡터
        normal = model[:3]
        normal_normalized = normal / np.linalg.norm(normal)
        
        # 수평면 판단 (Z축과 거의 평행)
        z_alignment = abs(np.dot(normal_normalized, [0, 0, 1]))
        
        # 수직면 판단 (XY 평면과 거의 수직)
        xy_alignment = np.sqrt(normal_normalized[0]**2 + normal_normalized[1]**2)
        
        # 평면 크기 (점 개수)
        size = len(plane.points)
        
        # 분류 로직
        if z_alignment > 0.9:  # Z축과 거의 평행 (수평면)
            classification['horizontal'].append((i, plane, model))
            plane_labels.append('horizontal')
            print(f"평면 {i+1}: 수평면 (바닥/천장) - Z축과의 정렬도: {z_alignment:.4f}")
            
        elif xy_alignment > 0.9:  # XY 평면과 거의 수직 (수직면)
            # 크기에 따라 주요 수직면과 보조 수직면 구분
            if size > len(planes[i].points) * 0.2:  # 평균보다 큰 경우
                classification['vertical_main'].append((i, plane, model))
                plane_labels.append('vertical_main')
                print(f"평면 {i+1}: 주요 수직면 (기둥) - XY 평면과의 정렬도: {xy_alignment:.4f}")
            else:
                classification['vertical_secondary'].append((i, plane, model))
                plane_labels.append('vertical_secondary')
                print(f"평면 {i+1}: 보조 수직면 - XY 평면과의 정렬도: {xy_alignment:.4f}")
                
        elif 0.3 < z_alignment < 0.85:  # 사선면 (브레이싱 등)
            classification['diagonal'].append((i, plane, model))
            plane_labels.append('diagonal')
            print(f"평면 {i+1}: 사선면 (브레이싱) - Z축과의 정렬도: {z_alignment:.4f}")
            
        else:
            classification['unknown'].append((i, plane, model))
            plane_labels.append('unknown')
            print(f"평면 {i+1}: 미분류 - Z축과의 정렬도: {z_alignment:.4f}, XY 평면과의 정렬도: {xy_alignment:.4f}")
    
    # 분류 요약
    print("\n분류 요약:")
    print(f"- 수평면 (바닥/천장): {len(classification['horizontal'])}개")
    print(f"- 주요 수직면 (기둥): {len(classification['vertical_main'])}개")
    print(f"- 보조 수직면: {len(classification['vertical_secondary'])}개")
    print(f"- 사선면 (브레이싱): {len(classification['diagonal'])}개")
    print(f"- 미분류: {len(classification['unknown'])}개")
    
    return classification, plane_labels

def deep_copy_point_cloud(pcd):
    """포인트 클라우드의 깊은 복사본을 생성합니다."""
    points = np.asarray(pcd.points)
    
    # 색상 정보가 있는지 확인
    has_colors = hasattr(pcd, 'colors') and len(pcd.colors) > 0
    
    # 법선 벡터 정보가 있는지 확인
    has_normals = hasattr(pcd, 'normals') and len(pcd.normals) > 0
    
    # 새로운 포인트 클라우드 객체 생성
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(points)
    
    # 색상 정보 복사
    if has_colors:
        colors = np.asarray(pcd.colors)
        new_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 법선 벡터 정보 복사
    if has_normals:
        normals = np.asarray(pcd.normals)
        new_pcd.normals = o3d.utility.Vector3dVector(normals)
    
    return new_pcd

def color_planes_by_classification(planes, labels):
    """분류에 따라 평면들의 색상을 지정합니다."""
    # 분류별 색상 정의
    classification_colors = {
        'horizontal': [0, 0.8, 0],      # 수평면: 초록색
        'vertical_main': [0.9, 0, 0],   # 주요 수직면: 빨간색
        'vertical_secondary': [0, 0, 0.9], # 보조 수직면: 파란색
        'diagonal': [0.9, 0.7, 0],      # 사선면: 주황색
        'unknown': [0.5, 0.5, 0.5],     # 미분류: 회색
        'other': [0.7, 0.7, 0.7]        # 기타 포인트: 옅은 회색
    }
    
    # 각 평면의 색상 지정
    colored_planes = []
    for i, (plane, label) in enumerate(zip(planes, labels)):
        colored_plane = deep_copy_point_cloud(plane)
        if label in classification_colors:
            colored_plane.paint_uniform_color(classification_colors[label])
        else:
            colored_plane.paint_uniform_color(classification_colors['other'])
        colored_planes.append(colored_plane)
    
    return colored_planes

def visualize_classified_planes(planes, labels, window_name="분류된 철골구조물 면"):
    """분류된 평면들을 시각화합니다."""
    print("분류된 평면들을 시각화합니다. 창을 닫으면 계속 진행됩니다.")
    
    # 분류별로 색상 지정
    colored_planes = color_planes_by_classification(planes, labels)
    
    # 좌표축 생성
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    # 모든 평면과 좌표축을 포함하는 리스트 생성
    geometries = colored_planes + [coordinate_frame]
    
    # 시각화
    o3d.visualization.draw_geometries(geometries, window_name=window_name)
    
    return colored_planes

def main():
    # 포인트 클라우드 파일 경로
    input_path = "results/3_pointcloud_filtered_2m.ply"
    if not os.path.exists(input_path):
        input_path = "3_pointcloud_filtered_2.0m.ply"
        if not os.path.exists(input_path):
            print("3_pointcloud.ply 파일을 찾을 수 없습니다.")
            return
    
    # 출력 디렉토리
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 포인트 클라우드 로드
    pcd = load_point_cloud(input_path)
    original_pcd = deep_copy_point_cloud(pcd)  # 원본 복사
    
    # 2. 포인트 클라우드 전처리
    preprocessed_pcd = preprocess_point_cloud(pcd, voxel_size=0.02)
    
    # 3. 평면 감지 파라미터 설정
    auto_adjust = input("임계값을 자동으로 조정하시겠습니까? (y/n): ").lower() == 'y'
    
    if auto_adjust:
        target_planes = int(input("목표 평면 개수를 입력하세요 (기본값: 5): ") or "5")
        distance_threshold = auto_adjust_threshold(
            preprocessed_pcd, 
            initial_threshold=0.02, 
            target_planes=target_planes
        )
    else:
        distance_threshold = float(input("평면 감지 임계값을 입력하세요 (기본값: 0.02m): ") or "0.02")
    
    min_plane_size = int(input("최소 평면 크기를 입력하세요 (기본값: 100 포인트): ") or "100")
    
    # 평면 감지 실행
    planes, colors, models = detect_planes(
        preprocessed_pcd, 
        distance_threshold=distance_threshold,
        min_plane_size=min_plane_size
    )
    
    # 4. 감지된 평면 분석
    if len(models) > 0:
        analyze_planes(planes[:-1], models)  # 마지막 '기타' 부분은 제외
        
        # 5. 철골구조물 평면 분류
        classify_option = input("철골구조물의 각 면으로 분류하시겠습니까? (y/n): ").lower() == 'y'
        if classify_option:
            classification, plane_labels = classify_steel_structure_planes(planes[:-1], models)
            
            # 마지막 '기타' 평면을 위한 레이블 추가
            if len(planes) > len(models):
                plane_labels.append('other')
                
            # 분류별 색상으로 시각화
            colored_planes = visualize_classified_planes(planes, plane_labels, window_name="분류된 철골 구조물 면")
            
            # 분류된 평면 저장
            save_classified = input("분류된 철골구조물 면을 저장하시겠습니까? (y/n): ").lower() == 'y'
            if save_classified:
                classified_dir = os.path.join(output_dir, "classified_planes")
                os.makedirs(classified_dir, exist_ok=True)
                
                # 분류별로 저장
                for category, planes_list in classification.items():
                    if planes_list:
                        # 카테고리별 디렉토리 생성
                        category_dir = os.path.join(classified_dir, category)
                        os.makedirs(category_dir, exist_ok=True)
                        
                        for idx, plane, _ in planes_list:
                            output_path = os.path.join(category_dir, f"plane_{idx+1}.ply")
                            o3d.io.write_point_cloud(output_path, plane)
                            print(f"{category} 평면 {idx+1}이 {output_path}에 저장되었습니다.")
    
    # 6. 원래 평면 시각화 (분류를 수행하지 않았거나 사용자가 원할 경우)
    if not classify_option or input("원래 감지된 평면도 시각화하시겠습니까? (y/n): ").lower() == 'y':
        visualize_planes(planes, colors, window_name="감지된 철골 구조물 면")
    
    # 7. 원본과 결과 비교 시각화
    compare_result = input("원본 포인트 클라우드와 결과를 비교 시각화하시겠습니까? (y/n): ").lower() == 'y'
    if compare_result:
        visualize_result_comparison(original_pcd, planes, output_dir)
    
    # 8. 각 평면 개별 저장 (분류하지 않은 원본 평면)
    if not classify_option:
        save_planes = input("감지된 각 평면을 개별 파일로 저장하시겠습니까? (y/n): ").lower() == 'y'
        if save_planes:
            save_plane_clouds(planes)

if __name__ == "__main__":
    main() 