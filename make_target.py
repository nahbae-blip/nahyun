import os
from PIL import Image, ImageFilter

source_dir = './Source_Dataset'
target_dir = './Target_Dataset'

# Source 폴더 안의 클래스 폴더명(training_fake, training_real) 가져오기
for class_name in os.listdir(source_dir):
    src_class_dir = os.path.join(source_dir, class_name)
    
    if os.path.isdir(src_class_dir):
        tgt_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(tgt_class_dir, exist_ok=True)
        
        # 각 폴더에서 50장만 가져옴
        images = os.listdir(src_class_dir)[:50]
        
        for img_name in images:
            src_path = os.path.join(src_class_dir, img_name)
            tgt_path = os.path.join(tgt_class_dir, img_name)
            
            try:
                # 논문의 Target 도메인 환경(열화) 구현
                img = Image.open(src_path).convert('RGB')
                img = img.filter(ImageFilter.GaussianBlur(radius=1.5)) # 블러
                img.save(tgt_path, 'JPEG', quality=30) # 화질 저하
            except Exception as e:
                pass

print("🎉 Target_Dataset 생성 완료! 이제 2단계 코드를 수정해 주세요.")
