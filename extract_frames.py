import cv2
import os
import shutil
import numpy as np

def extract_frames(video_path, output_dir, interval=2):
    # 1. 파일이 실제로 존재하는지 먼저 확인
    if not os.path.exists(video_path):
        print(f"🚨 에러: 비디오 파일이 없습니다!\n경로: {video_path}")
        print("💡 해당 폴더에 'DeepFakes_video.mp4' 파일이 진짜로 있는지 확인해 주세요.")
        return

    # 2. 저장할 폴더 만들기
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 '{output_dir}' 폴더를 생성했습니다.")

    # 3. 한글 경로 우회: 비디오 파일을 C:\Temp로 잠시 복사해서 사용
    temp_dir = r"C:\Temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
    shutil.copyfile(video_path, temp_video_path)
    
    # 안전한 임시 경로에서 비디오 열기
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        print(f"🚨 Error: 임시 경로에서도 비디오를 열 수 없습니다. 비디오 파일이 손상되었을 수 있습니다.")
        return

    print(f"🎬 비디오에서 프레임 추출을 시작합니다... (매 {interval}프레임당 1장)")
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # interval 주기마다 저장
        if frame_idx % interval == 0:
            frame_name = f"frame_{saved_count:05d}.jpg"
            save_path = os.path.join(output_dir, frame_name)
            
            # 4. 한글 경로 이미지 저장 (cv2.imwrite 대신 numpy로 우회)
            result, encoded_img = cv2.imencode('.jpg', frame)
            if result:
                with open(save_path, mode='w+b') as f:
                    encoded_img.tofile(f)
            saved_count += 1
            
        frame_idx += 1

    cap.release()
    
    # 작업이 끝나면 컴퓨터 용량 확보를 위해 임시 비디오 파일 삭제
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
        
    print(f"🎉 완료: 총 {saved_count}개의 프레임 이미지가 '{output_dir}'에 저장되었습니다.")

# ----------------------------------------------------
# 사용 예시
# ----------------------------------------------------
video_file = r"C:\Users\배나현\Desktop\code실습\real\original.mp4" 
output_folder = r"C:\Users\배나현\Desktop\code실습\real_frames" 

extract_frames(video_file, output_folder, interval=2)
# 사용 예시 (Fake 영상 추출용으로 수정)
# ----------------------------------------------------
# 1. 가짜 영상 파일이 있는 정확한 경로
video_file = r"C:\Users\배나현\Desktop\code실습\DeepFaceLab\result.mp4" 

# 2. 프레임 이미지들이 저장될 경로
output_folder = r"C:\Users\배나현\Desktop\code실습\fake_frames" 

extract_frames(video_file, output_folder, interval=2)
