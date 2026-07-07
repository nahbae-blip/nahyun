import cv2
import os
import mediapipe as mp

def crop_and_resize_face(image_dir, output_dir, target_size=(256, 256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # MediaPipe 얼굴 탐지 모델 로드
    mp_face_detection = mp.solutions.face_detection
    
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        for file_name in os.listdir(image_dir):
            if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(image_dir, file_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            h, w, _ = image.shape
            
            # MediaPipe는 RGB 이미지를 입력으로 받음 (OpenCV는 BGR)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_image)
            
            if results.detections:
                # 가장 신뢰도가 높은 첫 번째 얼굴 선택
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                # 상대 좌표를 절대 픽셀 좌표로 변환
                xmin = int(bbox.xmin * w)
                ymin = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # 바운딩 박스가 이미지 범위를 벗어나지 않도록 보정
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w, xmin + width), min(h, ymin + height)
                
                # 얼굴 영역 크롭
                face_crop = image[ymin:ymax, xmin:xmax]
                
                if face_crop.size == 0:
                    continue
                    
                # 모델 입력 크기(256x256)로 리사이징 (정규화)
                resized_face = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
                
                # 최종 저장
                output_path = os.path.join(output_dir, f"face_{file_name}")
                cv2.imwrite(output_path, resized_face)

    print(f"얼굴 크롭 및 정규화 완료! 저장 경로: {output_dir}")

# 사용 예시
input_frames_folder = "path/to/output/frames"
final_dataset_folder = "path/to/final/dataset"
crop_and_resize_face(input_frames_folder, final_dataset_folder)
