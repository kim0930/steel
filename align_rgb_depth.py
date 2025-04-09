import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import open3d as o3d

def load_images(rgb_path, depth_path):
    """RGB 이미지와 뎁스 이미지를 로드합니다."""
    rgb_img = cv2.imread(rgb_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR로 로드하므로 RGB로 변환
    
    # 뎁스 이미지 로드 (그레이스케일로)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth_img is None:
        # 일반 이미지로 로드 시도
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    
    return rgb_img, depth_img

def normalize_depth(depth_img):
    """뎁스 이미지를 시각화를 위해 정규화합니다."""
    min_val = np.min(depth_img)
    max_val = np.max(depth_img)
    if max_val > min_val:
        normalized = (depth_img - min_val) * 255.0 / (max_val - min_val)
        return normalized.astype(np.uint8)
    return depth_img

def resize_to_match(img1, img2):
    """두 이미지의 크기를 일치시킵니다. 더 큰 이미지를 작은 이미지 크기에 맞춥니다."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if h1 * w1 > h2 * w2:  # img1이 더 큼
        img1_resized = cv2.resize(img1, (w2, h2))
        return img1_resized, img2
    else:  # img2가 더 큼
        img2_resized = cv2.resize(img2, (w1, h1))
        return img1, img2_resized

def create_colored_depth(depth_img):
    """뎁스 이미지에 색상을 적용하여 시각화합니다."""
    colored_depth = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
    colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)
    return colored_depth

def align_images(rgb_img, depth_img):
    """RGB 이미지와 뎁스 이미지를 정합합니다."""
    # 이미지 크기 맞추기
    rgb_resized, depth_resized = resize_to_match(rgb_img, depth_img)
    
    # 뎁스 이미지 정규화 및 색상화
    normalized_depth = normalize_depth(depth_resized)
    colored_depth = create_colored_depth(normalized_depth)
    
    # 알파 블렌딩을 통한 정합 (50:50)
    alpha = 0.5
    aligned_img = cv2.addWeighted(rgb_resized, alpha, colored_depth, 1-alpha, 0)
    
    return rgb_resized, depth_resized, normalized_depth, colored_depth, aligned_img

def visualize_results(rgb, depth, colored_depth, aligned, output_path=None):
    """결과 이미지들을 시각화합니다."""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.title('RGB 이미지')
    plt.imshow(rgb)
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title('뎁스 이미지')
    plt.imshow(depth, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title('컬러 뎁스 이미지')
    plt.imshow(colored_depth)
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title('정합된 이미지')
    plt.imshow(aligned)
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"결과가 {output_path}에 저장되었습니다.")
    
    plt.show()

def create_point_cloud(rgb_img, depth_img, focal_length=500, scale_factor=1.0):
    """RGB 이미지와 뎁스 이미지로부터 3D 포인트 클라우드를 생성합니다.
    
    Parameters:
    - rgb_img: RGB 컬러 이미지
    - depth_img: 뎁스 이미지
    - focal_length: 카메라의 초점 거리 (Intel RealSense의 경우 약 500이 적절)
    - scale_factor: 깊이값 스케일 팩터
    
    Returns:
    - point_cloud: Open3D 포인트 클라우드 객체
    """
    # 이미지 크기가 다르면 맞춤
    if rgb_img.shape[:2] != depth_img.shape[:2]:
        rgb_img, depth_img = resize_to_match(rgb_img, depth_img)
    
    # 깊이 이미지 정규화
    if np.max(depth_img) > 255:
        # 16비트 깊이 이미지인 경우 스케일 조정
        depth_img = depth_img.astype(np.float32) / 65535.0 * 255.0
    
    height, width = depth_img.shape[:2]
    
    # 카메라 중심점 (이미지 중앙으로 가정)
    cx, cy = width / 2, height / 2
    
    # 3D 포인트 좌표 계산을 위한 인덱스 생성
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    y_indices, x_indices = indices[:, :, 0], indices[:, :, 1]
    
    # 깊이 값으로부터 Z 좌표 계산 (스케일 적용)
    z = depth_img.astype(np.float32) * scale_factor
    
    # X, Y 좌표 계산
    x = (x_indices - cx) * z / focal_length
    y = (y_indices - cy) * z / focal_length
    
    # 유효한 깊이 값을 가진 포인트만 선택 (0이 아닌 값)
    mask = z > 0
    
    # 3D 좌표 배열 생성
    points = np.zeros((np.sum(mask), 3), dtype=np.float32)
    points[:, 0] = x[mask]
    points[:, 1] = y[mask]
    points[:, 2] = z[mask]
    
    # RGB 색상 배열 생성
    colors = np.zeros((np.sum(mask), 3), dtype=np.float32)
    colors[:, 0] = rgb_img[mask, 0] / 255.0
    colors[:, 1] = rgb_img[mask, 1] / 255.0
    colors[:, 2] = rgb_img[mask, 2] / 255.0
    
    # Open3D 포인트 클라우드 생성
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    return point_cloud

def visualize_point_cloud(point_cloud, window_name="포인트 클라우드 시각화"):
    """Open3D를 사용하여 포인트 클라우드를 시각화합니다."""
    # 좌표축 생성
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    # 시각화
    o3d.visualization.draw_geometries([point_cloud, coordinate_frame], window_name=window_name)

def save_point_cloud(point_cloud, output_path):
    """포인트 클라우드를 파일로 저장합니다."""
    o3d.io.write_point_cloud(output_path, point_cloud)
    print(f"포인트 클라우드가 {output_path}에 저장되었습니다.")

def create_mesh_from_point_cloud(point_cloud, depth_trunc=5.0, voxel_size=0.05):
    """포인트 클라우드로부터 메쉬를 생성합니다.
    
    Parameters:
    - point_cloud: Open3D 포인트 클라우드 객체
    - depth_trunc: 깊이 절단 값
    - voxel_size: 복셀 크기
    
    Returns:
    - mesh: 생성된 메쉬
    """
    # 포인트 클라우드 다운샘플링 (처리 속도 향상을 위함)
    downsampled = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    
    # 노이즈 제거
    downsampled, _ = downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 법선 벡터 계산
    downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    downsampled.orient_normals_towards_camera_location()
    
    # 포아송 표면 복원
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        downsampled, depth=9, width=0, scale=1.1, linear_fit=False)
    
    # 낮은 밀도의 삼각형 제거
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # 메쉬 다듬기
    mesh = mesh.filter_smooth_simple(number_of_iterations=5)
    
    return mesh

def visualize_3d(geometries, window_name="3D 시각화"):
    """Open3D를 사용하여 3D 객체들을 시각화합니다."""
    # 좌표축 생성
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    all_geometries = geometries + [coordinate_frame]
    
    # 시각화
    o3d.visualization.draw_geometries(all_geometries, window_name=window_name)

def process_image_pair(rgb_path, depth_path, output_dir="results"):
    """한 쌍의 RGB와 뎁스 이미지를 처리합니다."""
    print(f"처리 중: {rgb_path} & {depth_path}")
    
    # 이미지 로드
    rgb_img, depth_img = load_images(rgb_path, depth_path)
    
    # 정합 수행
    rgb_resized, depth_resized, normalized_depth, colored_depth, aligned_img = align_images(rgb_img, depth_img)
    
    # 결과 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일 이름 추출
    base_name = os.path.splitext(os.path.basename(rgb_path))[0].replace("_rgb", "")
    
    # 결과 저장 경로
    output_path = os.path.join(output_dir, f"{base_name}_aligned.png")
    aligned_output_path = os.path.join(output_dir, f"{base_name}_aligned_only.png")
    point_cloud_path = os.path.join(output_dir, f"{base_name}_pointcloud.ply")
    mesh_path = os.path.join(output_dir, f"{base_name}_mesh.ply")
    
    # 결과 시각화 및 저장
    visualize_results(rgb_resized, normalized_depth, colored_depth, aligned_img, output_path)
    
    # 정합된 이미지만 저장
    plt.figure(figsize=(10, 8))
    plt.imshow(aligned_img)
    plt.axis('off')
    plt.savefig(aligned_output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"정합된 이미지가 {aligned_output_path}에 저장되었습니다.")
    
    # 3D 포인트 클라우드 생성 및 시각화
    print("3D 포인트 클라우드 생성 중...")
    point_cloud = create_point_cloud(rgb_resized, depth_resized)
    
    # 포인트 클라우드 저장
    save_point_cloud(point_cloud, point_cloud_path)
    
    # 포인트 클라우드 시각화
    print("3D 포인트 클라우드를 시각화합니다...")
    visualize_point_cloud(point_cloud, window_name=f"{base_name} - 포인트 클라우드")
    
    # # 3D 메쉬 생성 시도
    # try:
    #     print("3D 메쉬 생성 중... (시간이 다소 소요될 수 있습니다)")
    #     mesh = create_mesh_from_point_cloud(point_cloud)
        
    #     # 메쉬 저장
    #     o3d.io.write_triangle_mesh(mesh_path, mesh)
    #     print(f"3D 메쉬가 {mesh_path}에 저장되었습니다.")
        
    #     # 메쉬 시각화
    #     print("3D 메쉬를 시각화합니다...")
    #     visualize_3d([mesh], window_name=f"{base_name} - 3D 메쉬")
        
    #     return rgb_resized, depth_resized, normalized_depth, colored_depth, aligned_img, point_cloud, mesh
    # except Exception as e:
    #     print(f"메쉬 생성 중 오류 발생: {str(e)}")
    #     return rgb_resized, depth_resized, normalized_depth, colored_depth, aligned_img, point_cloud, None

def main():
    # 디렉토리에서 모든 이미지 쌍 처리
    image_pairs = [
        # ("1_rgb.png", "1_depth.png"),
        ("3_rgb.png", "3_depth.png")
    ]
    
    for rgb_file, depth_file in image_pairs:
        process_image_pair(rgb_file, depth_file)

if __name__ == "__main__":
    main() 