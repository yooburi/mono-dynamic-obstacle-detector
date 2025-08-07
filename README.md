# ROS 기반 YOLO 동적/정적 장애물 분류기 (ROS YOLO Dynamic Obstacle Detector)

이 ROS 패키지는 카메라 이미지로부터 객체를 탐지 및 추적하고, 카메라의 자체 움직임(Ego-motion)을 보상하여 최종적으로 해당 객체가 **동적인지(Dynamic)** 또는 **정적인지(Static)** 판단하는 노드를 제공합니다.

## 주요 기능

- **객체 탐지 및 추적**: `ultralytics YOLOv8` 모델을 사용하여 실시간으로 객체를 탐지하고 ID를 부여하여 추적합니다.
- **Ego-motion 보상**: 배경 특징점의 Optical Flow를 계산하여 카메라의 움직임을 추정하고, 이를 보상합니다.
- **동적/정적 분류**: Ego-motion 보상 후에도 독립적인 움직임이 관찰되는 객체를 '동적 장애물'로 분류합니다.
- **ROS 통합**: ROS 토픽을 통해 이미지 입력을 받고, 추적 결과 및 동적 장애물 여부를 다른 노드에 발행합니다.
- **디버그 시각화**: 원본 이미지에 바운딩 박스, 추적 ID, 예측 위치, 실제 위치 등을 시각화하여 디버깅용 이미지 토픽을 발행합니다.

## 동작 원리

1.  **객체 추적**: YOLO 모델이 입력 이미지에서 객체들을 탐지하고 각 객체에 고유 ID를 할당하여 추적을 시작합니다.
2.  **배경 마스킹**: 탐지된 객체 영역을 제외한 배경(background) 마스크를 생성합니다.
3.  **배경 특징점 추출**: 배경 영역에서만 `cv2.goodFeaturesToTrack`을 사용하여 추적할 특징점을 추출합니다.
4.  **Ego-motion 추정**: 이전 프레임과 현재 프레임의 배경 특징점들을 `cv2.calcOpticalFlowPyrLK` (Lucas-Kanade Optical Flow)를 이용해 추적하고, 이를 기반으로 `cv2.findHomography`를 통해 카메라의 움직임(Ego-motion)을 나타내는 Homography 행렬을 계산합니다.
5.  **움직임 오차 계산**: 추적 중인 각 객체에 대해, 이전 프레임 위치에 Homography 행렬을 적용하여 카메라 움직임만으로 예상되는 현재 위치를 계산합니다.
6.  **동적/정적 판단**: '예상 위치'와 YOLO로 추적된 '실제 현재 위치' 사이의 거리(Motion Error)를 계산합니다. 이 오차가 설정된 임계값(`MOTION_THRESHOLD_PX`)을 초과하면 해당 객체는 스스로 움직이는 **동적 장애물**로 판단합니다.

## 사전 요구사항

- ROS (Melodic, Noetic 등)
- Python 3
- PyTorch
- `ultralytics`
- `opencv-python`
- `numpy`
- `yolo_msgs` (사용자 정의 메시지 패키지)

```bash
pip install ultralytics
```

## 실행 방법

1.  **YOLO 모델 준비**: 학습된 YOLO 모델 파일 (`.pt`)을 준비합니다.
2.  **Launch 파일 작성**: 아래 예시와 같이 `roslaunch` 파일을 작성하여 파라미터를 설정하고 노드를 실행합니다.

    ```xml
    <!-- yolo_track.launch -->
    <launch>
        <node pkg="yolo_ros_ros1" type="yolo_track_node.py" name="yolo_track_node" output="screen">
            <!-- 실행에 필요한 파라미터 -->
            <param name="image_topic" value="/usb_cam/image_raw" />
            <param name="model_path" value="$(find yolo_ros_ros1)/models/best.pt" />
            <param name="imgsz_width" value="640" />
            <param name="imgsz_height" value="360" />
            
            <!-- 탐지/추적 파라미터 -->
            <param name="threshold" value="0.5" />
            <param name="iou" value="0.5" />
            
            <!-- 동적/정적 판단 파라미터 -->
            <param name="bbox_area_threshold" value="4500" /> <!-- 이 넓이 이상인 객체만 동적 판단 수행 -->
            <param name="MOTION_THRESHOLD_PX" value="10.0" /> <!-- Ego-motion 보상 후 이 픽셀 이상 움직이면 동적으로 판단 -->
        </node>
    </launch>
    ```

3.  **Launch 파일 실행**:
    ```bash
    roslaunch yolo_ros_ros1 yolo_track.launch
    ```

## ROS API

#### 구독 (Subscribed Topics)

-   **`~image_topic`** (`sensor_msgs/Image`)
    -   입력 비디오 스트림 이미지.
-   **`/steering_angle`** (`std_msgs/Float32`)
    -   차량의 현재 조향각. 직진에 가까운 상황에서만 동적 판단의 신뢰도를 높이기 위해 사용됩니다.

#### 발행 (Published Topics)

-   **`/tracking`** (`yolo_msgs/DetectionArray`)
    -   추적된 모든 객체의 정보 (ID, 클래스, 바운딩 박스, 동적 여부 포함된 라벨) 배열.
-   **`/dbg_image`** (`sensor_msgs/Image`)
    -   추적 결과가 시각화된 디버그용 이미지.
-   **`/cone_info`** (`std_msgs/String`)
    -   탐지된 콘의 정보를 요약한 문자열.
-   **`/motion_error_info`** (`std_msgs/String`)
    -   각 객체의 움직임 오차(Motion Error) 값 정보.
-   **`/dynamic_obstacle`** (`std_msgs/Bool`)
    -   동적 장애물이 하나라도 감지되었는지 여부 (True/False).
