import cv2
import os
import mediapipe as mp

def process_videos(video_folder, output_folder, target_size=(256, 256)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # MediaPipe 얼굴 탐지 모델 초기화
    mp_face_detection = mp.solutions.face_detection
    
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        
        # 비디오 폴더 안의 모든 파일 순회
        for video_name in os.listdir(video_folder):
            if not video_name.endswith('.mp4'):
                continue
                
            video_path = os.path.join(video_folder, video_name)
            cap = cv2.VideoCapture(video_path)
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 10프레임당 1장씩만 추출 (시간 단축 및 데이터 중복 방지)
                if frame_count % 10 == 0:
                    h, w, _ = frame.shape
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb_frame)
                    
                    if results.detections:
                        # 첫 번째 얼굴 추출
                        detection = results.detections[0]
                        bbox = detection.location_data.relative_bounding_box
                        
                        xmin = int(bbox.xmin * w)
                        ymin = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # 범위 보정
                        xmin, ymin = max(0, xmin), max(0, ymin)
                        xmax, ymax = min(w, xmin + width), min(h, ymin + height)
                        
                        face_crop = frame[ymin:ymax, xmin:xmax]
                        
                        if face_crop.size > 0:
                            # 256x256 크기로 리사이징
                            resized_face = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
                            
                            # 파일 저장 (비디오이름_프레임번호.jpg)
                            save_name = f"{video_name.split('.')[0]}_frame_{frame_count:04d}.jpg"
                            save_path = os.path.join(output_folder, save_name)
                            cv2.imwrite(save_path, resized_face)
                            
                frame_count += 1
                
            cap.release()
            print(f"완료: {video_name}")

# 실제 경로로 수정해 주세요
video_folder = r"C:\Users\ghdka\OneDrive\바탕 화면\논문 실습\dataset\sample_videos"
output_folder = r"C:\Users\ghdka\OneDrive\바탕 화면\논문 실습\dataset\processed_faces"

process_videos(video_folder, output_folder)
print("모든 비디오의 전처리가 완료되었습니다!")
