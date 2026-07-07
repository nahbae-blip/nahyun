import torch
from network.models import model_selection
from network.xception import xception # 이 줄이 빠져있어서 에러가 난 것입니다!
# 1. 모델 설정 (예: 'xception')
MODEL_NAME = 'xception' 

# 2. 모델 생성 및 가중치 로드
def load_my_model(model_path):
    # 이제 모델 생성과 가중치 로드는 models.py 안에서 다 처리됩니다.
    model, _, _, _, _ = model_selection(MODEL_NAME, num_out_classes=2)
    model.eval()
    return model

if __name__ == "__main__":
    MY_MODEL_PATH = r"C:\Users\배나현\Desktop\code실습\models\xception-b5690688.pth"
    model = load_my_model(MY_MODEL_PATH)
    print("모델 로드 성공!")
