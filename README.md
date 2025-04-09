# RGB-D 이미지 처리 및 3D 시각화 도구

이 프로젝트는 Intel RealSense 뎁스 카메라로 촬영한 철골구조물의 RGB 이미지와 뎁스 이미지를 처리하고 시각화하는 도구입니다.

## 주요 기능

1. RGB 이미지와 뎁스 이미지 정합(alignment)
2. 3D 포인트 클라우드 생성 및 시각화
3. 3D 메쉬 생성 및 시각화

## 설치 방법

필요한 라이브러리를 설치합니다:

```bash
pip install -r requirements.txt
```

## 사용 방법

1. RGB 이미지와 뎁스 이미지를 준비합니다. (예: `1_rgb.png`, `1_depth.png`)
2. 다음 명령어로 프로그램을 실행합니다:

```bash
python align_rgb_depth.py
```

3. 결과물은 `results` 폴더에 저장됩니다:
   - `*_aligned.png`: 정합 시각화 결과
   - `*_aligned_only.png`: 정합된 이미지
   - `*_pointcloud.ply`: 3D 포인트 클라우드 파일
   - `*_mesh.ply`: 3D 메쉬 파일

## 결과물 확인

3D 결과물 (.ply 파일)은 다음과 같은 도구로 확인할 수 있습니다:
- [MeshLab](https://www.meshlab.net/)
- [CloudCompare](https://www.danielgm.net/cc/)
- [Blender](https://www.blender.org/)

## 코드 설명

### 주요 함수

- `load_images`: RGB 및 뎁스 이미지 로드
- `align_images`: RGB 및 뎁스 이미지 정합
- `create_point_cloud`: 3D 포인트 클라우드 생성
- `create_mesh_from_point_cloud`: 포인트 클라우드에서 3D 메쉬 생성
- `visualize_point_cloud`: 포인트 클라우드 시각화
- `visualize_3d`: 3D 객체(메쉬 등) 시각화

## 참고사항

- 메쉬 생성 과정은 연산 비용이 높아 시간이 오래 걸릴 수 있습니다.
- 뎁스 이미지의 품질에 따라 3D 재구성 결과가 달라질 수 있습니다.
- 기본 초점 거리(focal_length)는 500으로 설정되어 있습니다. 실제 카메라 사양에 맞게 조정이 필요할 수 있습니다. 