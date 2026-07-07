import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os

# ==========================================
# 0. 환경 설정
# ==========================================
# GPU 사용 가능 여부 확인 (DirectML 환경이시면 torch_directml.device() 사용)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ 현재 사용 중인 연산 장치: {device}")

# ★ 본인이 저장한 Source 데이터셋 경로로 수정하세요!
SOURCE_DATA_DIR = './Source_Dataset' 

# ==========================================
# 1. 모델 레이어 교체용 유틸리티 (BatchNorm -> GroupNorm) - 수정됨!
# ==========================================
def convert_bn_to_gn(module, target_groups=32):
    """배치사이즈에 독립적인 성능을 내기 위해 GroupNorm 적용 (논문 요구사항)"""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features
            
            # 핵심 해결책: channels가 나누어 떨어질 때까지 그룹 수를 줄임
            groups = target_groups
            while channels % groups != 0:
                groups -= 1
                
            setattr(module, name, nn.GroupNorm(num_groups=groups, num_channels=channels))
        else:
            convert_bn_to_gn(child, target_groups)

# ==========================================
# 2. T-GD 모델 정의 (EfficientNet-B0) - 수정됨!
# ==========================================
class TGDEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super(TGDEfficientNet, self).__init__()
        # 경고 메시지 안 뜨도록 최신 파이토치 문법으로 수정
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # BN -> GroupNorm 교체
        convert_bn_to_gn(self.backbone)
        
        # FC Layer 수정 (과대적합 방지용 Dropout 포함)
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
# ==========================================
# 3. 데이터 로더 준비
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)), # EfficientNet 권장 사이즈
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

source_dataset = datasets.ImageFolder(root=SOURCE_DATA_DIR, transform=transform)
source_loader = DataLoader(source_dataset, batch_size=16, shuffle=True)
print(f"📦 Source 데이터 로더 준비 완료! (총 {len(source_dataset)}장)")
print(f"클래스 매핑: {source_dataset.class_to_idx}")

# ==========================================
# 4. 1단계 사전 학습 루프 (Source Pre-training)
# ==========================================
if __name__ == '__main__':
    model = TGDEfficientNet().to(device)
    
    # 1단계는 일반적인 분류 학습이므로 CrossEntropy와 Adam 사용
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 3 # 초고속 실습을 위해 3 에포크만 진행
    
    print("🚀 [1단계] Source 데이터 사전 학습을 시작합니다!")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(source_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"  > [Epoch {epoch+1}, Batch {batch_idx}] Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(source_loader)
        epoch_acc = 100. * correct / total
        print(f"📈 Epoch [{epoch+1}/{epochs}] 완료 | 평균 손실: {epoch_loss:.4f} | 정확도: {epoch_acc:.2f}%")

    # 논문의 핵심: 이 사전 학습된 가중치를 보존하는 것!
    torch.save(model.state_dict(), 'tgd_source_pretrained.pth')
    print("🎉 [1단계 완료] 사전 학습된 모델 가중치가 'tgd_source_pretrained.pth'로 저장되었습니다!")
