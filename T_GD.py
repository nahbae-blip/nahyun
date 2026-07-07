import torch
import torch_directml
import torch.nn as nn
import torchvision.models as models
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch_directml.device()
print(f"현재 사용 중인 연산 장치: {device}")

# -----------------------------------------------------
# 1. 데이터 증강 (Albumentations 적용)
# 논문 명시: JPEG compression, Gaussian blur, random horizontal flip, Cutmix
# -----------------------------------------------------
# Albumentations는 Numpy 배열(H, W, C)을 입력으로 받습니다.
# Dataset의 __getitem__ 에서 image = np.array(image) 후 transform(image=image)['image'] 형태로 사용해야 합니다.
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.ImageCompression(quality_range=(60, 90), p=0.5), # JPEG Compression 추가
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def cutmix_data(x, y, alpha=1.0):
    """Cutmix 구현: 두 이미지를 패치 단위로 섞고 라벨도 비율에 맞춰 혼합"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    W, H = x.size()[2], x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y, y[index], lam

# -----------------------------------------------------
# 2. T-GD 모델 정의 (EfficientNet-B0 기반)
# -----------------------------------------------------
class TGDEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super(TGDEfficientNet, self).__init__()
        # 사전 학습된 EfficientNet-B0 로드
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Dropout 적용 (과대적합 방지)
        self.dropout = nn.Dropout(p=0.5)
        
        # FC Layer 수정 (딥페이크 이진 분류)
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(num_ftrs, num_classes)
        )
        
        # Pre-trained 가중치 보존을 위한 복사본 저장 ($L^{2}$-SP 계산용)
        self.pretrained_weights = {
            name: param.clone().detach() 
            for name, param in self.backbone.named_parameters() if 'classifier' not in name
        }

    def forward(self, x):
        return self.backbone(x)
    
# -----------------------------------------------------
# 3. L2-SP 규제 손실 함수
# -----------------------------------------------------
def l2_sp_loss(outputs, targets, model, alpha, beta):
    """
    outputs: 모델 예측값
    targets: 정답 라벨 (또는 Pseudo-label)
    model: 학습 중인 Student 모델
    alpha: Pre-trained 가중치 보존 규제 계수
    beta: FC Layer 정규화 계수
    """
    criterion = nn.CrossEntropyLoss()
    ce_loss = criterion(outputs, targets)
    
    sp_loss = 0.0 # 사전 학습 모델과의 가중치 차이
    l2_loss = 0.0 # FC 레이어 규제
    
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            if name in model.pretrained_weights:
                pretrained_w = model.pretrained_weights[name].to(param.device)
                sp_loss += torch.sum((param - pretrained_w) ** 2)
        else:
            l2_loss += torch.sum(param ** 2)
            
    # 최종 손실 함수: J + alpha * Ω_sp + beta * Ω_l2
    total_loss = ce_loss + (alpha / 2) * sp_loss + (beta / 2) * l2_loss
    return total_loss

# -----------------------------------------------------
# 4. Self-training (Domain Adaptation) 학습 흐름
# -----------------------------------------------------
def train_tgd_target_domain(teacher_model, student_model, target_dataloader, optimizer, device):
    """
    Target 데이터(예: 새로운 딥페이크 도메인)에 대해 Teacher가 생성한
    Pseudo-label을 기반으로 Student를 학습시킵니다.
    """
    teacher_model.eval()  # Teacher는 평가 모드 (가중치 고정)
    student_model.train() # Student는 학습 모드
    
    # 논문 권장: 낮은 학습률과 낮은 관성(momentum)
    # optimizer = torch.optim.SGD(student_model.parameters(), lr=0.001, momentum=0.5)
    
    for batch_idx, (inputs, _) in enumerate(target_dataloader):
        # Target 데이터셋의 실제 라벨(_)은 사용하지 않습니다 (Unsupervised Domain Adaptation)
        inputs = inputs.to(device)
        
        # 1. Teacher 모델을 활용한 Pseudo-label 생성 및 신뢰도 측정
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
            probabilities = torch.softmax(teacher_logits, dim=1)
            # 가장 확률이 높은 클래스를 가짜 정답(Pseudo-label)으로 사용
            confidences, pseudo_targets = torch.max(probabilities, dim=1) 
            
        # 2. Self-training 시 노이즈 주입 (과대적합 방지)
        noise = torch.randn_like(inputs) * 0.05
        inputs_noisy = inputs + noise
        
        # 3. Cutmix 적용 (Pseudo-label 기준)
        inputs_mixed, targets_a, targets_b, lam = cutmix_data(inputs_noisy, pseudo_targets)
        
        # 4. 동적 alpha, beta 조정
        mean_confidence = confidences.mean().item()
        # Teacher가 확신할수록 사전 학습 가중치 의존도(alpha)를 줄임
        raw_alpha = 0.1 * (1.0 - mean_confidence) 
        current_alpha = max(0.001, min(raw_alpha, 0.1)) # 너무 극단적인 값을 막기 위한 Clipping
        current_beta = 0.01
        
        # 5. Student 모델 학습
        optimizer.zero_grad()
        outputs = student_model(inputs_mixed)
        
        # Cutmix 비율(lam)을 반영한 최종 손실 계산
        loss_a = l2_sp_loss(outputs, targets_a, student_model, current_alpha, current_beta)
        loss_b = l2_sp_loss(outputs, targets_b, student_model, current_alpha, current_beta)
        loss = loss_a * lam + loss_b * (1. - lam)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Batch: {batch_idx}, Loss: {loss.item():.4f}, Alpha: {current_alpha:.4f}, Confidence: {mean_confidence:.4f}")
