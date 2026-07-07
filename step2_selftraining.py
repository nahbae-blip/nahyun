import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_DATA_DIR = './Target_Dataset'

# 1. 모델 레이어 교체용 유틸리티
def convert_bn_to_gn(module, target_groups=32):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features
            groups = target_groups
            while channels % groups != 0:
                groups -= 1
            setattr(module, name, nn.GroupNorm(num_groups=groups, num_channels=channels))
        else:
            convert_bn_to_gn(child, target_groups)

# 2. 모델 정의
class TGDEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super(TGDEfficientNet, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        convert_bn_to_gn(self.backbone)
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        
        # L2-SP용 Pre-trained 가중치 저장소
        self.pretrained_weights = {}

    def forward(self, x):
        return self.backbone(x)

# 3. L2-SP 손실 함수 (논문 수식 15)
def l2_sp_loss(outputs, pseudo_labels, model, alpha=0.1, beta=0.01):
    ce_loss = F.cross_entropy(outputs, pseudo_labels)
    sp_loss, l2_loss = 0.0, 0.0
    
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            if name in model.pretrained_weights:
                pretrained_w = model.pretrained_weights[name].to(param.device)
                sp_loss += torch.sum((param - pretrained_w) ** 2)
        else:
            l2_loss += torch.sum(param ** 2)
            
    return ce_loss + (alpha / 2) * sp_loss + (beta / 2) * l2_loss

if __name__ == '__main__':
    # ==========================================
    # [준비] Teacher와 Student 모델 세팅
    # ==========================================
    print("🚀 Teacher & Student 모델을 준비합니다...")
    teacher = TGDEfficientNet().to(device)
    student = TGDEfficientNet().to(device)
    
    # 1단계 가중치 불러오기
    weights = torch.load('tgd_source_pretrained.pth', weights_only=True)
    teacher.load_state_dict(weights)
    student.load_state_dict(weights)
    
    # Student 모델에 L2-SP 기준점이 될 가중치 저장
    student.pretrained_weights = {name: param.clone().detach() for name, param in teacher.named_parameters() if 'classifier' not in name}
    
    teacher.eval() # Teacher는 가중치 완전히 고정
    student.train()
    
    # ==========================================
    # [Self-training] 타겟 도메인 적응 (정답 없이!)
    # ==========================================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_dataset = datasets.ImageFolder(root=TARGET_DATA_DIR, transform=transform)
    target_loader = DataLoader(target_dataset, batch_size=16, shuffle=True)
    
    optimizer = optim.Adam(student.parameters(), lr=1e-5)
    
    print("🧠 원래 세팅(안정적인 임계값 방식)으로 Self-training을 시작합니다!")

    for epoch in range(3):
        for inputs, true_labels in target_loader:
            inputs = inputs.to(device)
            
            # 1. Teacher의 스스로 정답 생성 (Pseudo-labeling)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
                probs = F.softmax(teacher_outputs, dim=1)
                max_probs, pseudo_labels = torch.max(probs, dim=1)
            
            # 2. 원래의 안정적인 임계값(0.80)으로 복구
            mask = max_probs > 0.80
            
            if mask.sum() > 0:
                inputs_noisy = inputs[mask] + (torch.randn_like(inputs[mask]) * 0.05)
                outputs = student(inputs_noisy)
                
                # 복잡한 가중치 없이 순수한 L2-SP Loss 계산
                loss = l2_sp_loss(outputs, pseudo_labels[mask], student)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"  > Target 도메인 적응 Epoch [{epoch+1}/3] 완료")

    # ==========================================
    # [최종 평가] AUROC 계산
    # ==========================================
    print("📊 학습 완료! 최종 AUROC 성능을 평가합니다.")
    student.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, true_labels in target_loader:
            inputs = inputs.to(device)
            outputs = student(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1] # Fake일 확률
            
            all_labels.extend(true_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    auroc = roc_auc_score(all_labels, all_probs)
    print("========================================")
    print(f"🏆 최종 Target 도메인 AUROC 점수: {auroc * 100:.2f}%")
    print("========================================")
