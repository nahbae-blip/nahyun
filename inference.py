#얼굴이 잘라진 사진들을 모델에 넣어 가짜 확률을 확인하는 역할.
import torch
import torchvision.transforms as transforms
from PIL import Image
from main import load_my_model, MY_MODEL_PATH # 기존에 만든 코드 재활용

# 모델 로드
model = load_my_model(MY_MODEL_PATH)
model.eval()

# 이미지 전처리 설정 (Xception은 299x299 크기를 선호합니다)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0) # 배치 차원 추가 (1, 3, 299, 299)
    
    with torch.no_grad():
        output = model(input_tensor)
        # 딥페이크 탐지는 보통 Softmax 통과 전 로짓 값을 사용하거나 확률로 변환
        prob = torch.softmax(output, dim=1)
        
    return prob[0][1].item() # 1번 클래스(Fake)일 확률 반환

# 테스트
# 영상이 아니라, 추출된 이미지 중 하나를 지정해야 합니다!
test_img = r"C:\Users\배나현\Desktop\code실습\frames\frame_00000.jpg"
print(f"가짜일 확률: {predict_image(test_img):.4f}")
