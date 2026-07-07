import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
# 1. 연산 장치 설정 (AMD 내장 그래픽 VRAM 한계로 CPU 우선 사용)
device = torch.device('cpu')
print(f"🖥️ 현재 설정된 연산 장치: {device}")

# 2. 데이터 경로 설정 (1단계에서 만든 dataset 폴더)
#수정 전 data_dir = './dataset' 
# 수정 후
data_dir = './image_dataset'
# 3. 이미지 전처리 (Transform)
# ForensicTransfer 등 딥페이크 탐지 모델의 입력 기준에 맞게 조정
transform = transforms.Compose([
    transforms.Resize((256, 256)),        # 이미지를 256x256 크기로 통일
    transforms.ToTensor(),                # 이미지를 파이토치가 읽을 수 있는 숫자로 변환 (텐서화)
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 픽셀 값 정규화
])
# 4. 데이터셋 로드 (폴더명으로 자동 라벨링)
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# --- [추가] 데이터 분할 전략 ---
# 전체 데이터(561장)에서 50장은 사전 학습용, 20장은 재학습용으로 분리
total_size = len(dataset)
indices = torch.randperm(total_size).tolist()

# 50장: 사전 학습용, 20장: 재학습용
pretrain_indices = indices[:50]
target_indices = indices[50:70] 

# 사전 학습용 데이터셋 생성
pretrain_dataset = Subset(dataset, pretrain_indices)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=8, shuffle=True)

# 재학습용 로더도 미리 만들어 두세요
target_dataset = Subset(dataset, target_indices)
target_loader = DataLoader(target_dataset, batch_size=4, shuffle=True)
# ----------------------------

# 6. 연결 상태 테스트 확인
if __name__ == '__main__':
    print("====================================")
    print(f"✅ 총 불러온 전체 이미지 개수: {total_size}장")
    print(f"✅ 사전 학습용 이미지 개수: {len(pretrain_dataset)}장")
    print(f"✅ 재학습(Target)용 이미지 개수: {len(target_dataset)}장")
    print("====================================")
