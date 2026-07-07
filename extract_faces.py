import cv2
import os
import numpy as np
import shutil

# 1. 한글 경로 문제 우회: Haar Cascade 모델을 안전한 C 드라이브 임시 폴더로 복사해서 사용
temp_dir = r"C:\Temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

original_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
safe_cascade_path = os.path.join(temp_dir, 'haarcascade_frontalface_default.xml')

# 안전한 경로(Temp)에 파일이 없으면 복사
if not os.path.exists(safe_cascade_path):
    shutil.copyfile(original_cascade_path, safe_cascade_path)

# 에러가 나던 경로 대신, 안전한 경로에서 얼굴 인식기 로드
face_cascade = cv2.CascadeClassifier(safe_cascade_path)

def crop_faces(input_dir, output_dir):
    # 2. 폴더가 없으면 자동으로 생성
    if not os.path.exists(input_dir):
        print(f"🚨 알림: '{input_dir}' 폴더가 없어서 새로 생성했습니다.")
        print("💡 방금 만들어진 폴더 안에 원본 얼굴 사진들을 넣고 코드를 다시 실행해주세요!")
        os.makedirs(input_dir)
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"✅ '{input_dir}' 폴더에서 얼굴 추출을 시작합니다...")
    
    saved_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(input_dir, filename)
            
            # 3. 한글 경로 이미지 읽기 (cv2.imread 대신 numpy로 우회)
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 수정 후 (검출 정밀도 완화)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20))
            
            for i, (x, y, w, h) in enumerate(faces):
                face_img = img[y:y+h, x:x+w]
                save_path = os.path.join(output_dir, f"face_{i}_{filename}")
                
                # 4. 한글 경로 이미지 저장 (cv2.imwrite 대신 numpy로 우회)
                result, encoded_img = cv2.imencode('.jpg', face_img)
                if result:
                    with open(save_path, mode='w+b') as f:
                        encoded_img.tofile(f)
                saved_count += 1
                
    if saved_count > 0:
        print(f"🎉 성공! 총 {saved_count}개의 얼굴 이미지를 '{output_dir}'에 저장했습니다.")
    else:
        print("⚠️ 추출된 얼굴이 없습니다. 폴더에 이미지가 있는지 확인해주세요.")

# ----------------------------------------------------
# 실행 부분
# ----------------------------------------------------
input_folder = r"C:\Users\배나현\Desktop\code실습\real_frames"
output_folder = r"C:\Users\배나현\Desktop\code실습\real_faces"

crop_faces(input_folder, output_folder)



input_folder = r"C:\Users\배나현\Desktop\code실습\fake_frames"
output_folder = r"C:\Users\배나현\Desktop\code실습\fake_faces"

crop_faces(input_folder, output_folder)
