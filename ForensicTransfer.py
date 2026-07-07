import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import csv

# ==========================================
# 1. 환경 및 데이터 로더 세팅
# ==========================================
device = torch.device('cpu')
print(f"🖥️ 현재 사용 중인 연산 장치: {device}")



transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# 2. 데이터 로더 정의
data_dir = './image_dataset'
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

total_size = len(dataset)
indices = torch.randperm(total_size).tolist()
pretrain_indices = indices[:50]
target_indices = indices[50:70]

# 여기서 정의합니다!
pretrain_dataset = Subset(dataset, pretrain_indices)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=8, shuffle=True)

target_dataset = Subset(dataset, target_indices)
target_loader = DataLoader(target_dataset, batch_size=4, shuffle=True)
print(f"📦 데이터 로더 준비 완료! (총 {len(dataset)}장)")
# 데이터 로더 확인 코드
print(f"클래스 매핑 확인: {dataset.class_to_idx}")
# 만약 {'Fake': 0, 'Real': 1}이라면, 
# labels가 0일 때 'Fake' 특징을 마스킹하고 있으므로 논문과 반대입니다.
# ==========================================
# 2. ForensicTransfer 모델 정의
# ==========================================
class ForensicTransfer(nn.Module):
    def __init__(self):
        super(ForensicTransfer, self).__init__()
        self.encoder = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), nn.ReLU())
        ])
        self.decoder = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1), nn.Tanh())
        ])

    def selection_block(self, latent, labels):
        masked_latent = latent.clone()
        
        for i, label in enumerate(labels):
            # label 0이 '진짜'라고 가정할 때
            if label == 0: 
                # 진짜일 때는 가짜 특징맵(64:)을 0으로
                masked_latent[i, 64:, :, :] = 0.0 
                # 진짜 특징맵(:64)은 건드리지 않음 (값 유지)
            else:
                # 가짜일 때는 진짜 특징맵(:64)을 0으로
                masked_latent[i, :64, :, :] = 0.0 
                # 가짜 특징맵(64:)은 건드리지 않음 (값 유지)
                
        return masked_latent

    def forward(self, x, labels=None):
        for layer in self.encoder: x = layer(x)
        latent = x 
        decoder_input = self.selection_block(latent, labels) if labels is not None else latent
        out = decoder_input
        for layer in self.decoder: out = layer(out)
        return out, latent

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: init.constant_(m.bias, 0)

# ==========================================
# 3. 손실 함수 및 예측 함수
# ==========================================
def forensic_loss(reconstructed, original, latent, labels, lambda_weight=0.1):
    L_rec = torch.nn.functional.mse_loss(reconstructed, original)
    a0 = torch.sum(torch.abs(latent[:, :64, :, :]), dim=[1, 2, 3])
    a1 = torch.sum(torch.abs(latent[:, 64:, :, :]), dim=[1, 2, 3])
    L_act = 0.0
    for i, label in enumerate(labels):
        if label == 1: # Real (클래스 매핑 확인 결과에 따라 수정)
            # Real일 때는 a0를 1로, a1을 0으로 유도
            L_act += torch.abs(a0[i] - 1) + torch.abs(a1[i])
        else: # Fake (label 0)
            # Fake일 때는 a1을 1로, a0를 0으로 유도
            L_act += torch.abs(a1[i] - 1) + torch.abs(a0[i])
    return (L_act / len(labels)) + (lambda_weight * L_rec)


def predict_deepfake(latent):
    a0 = torch.sum(torch.abs(latent[:, :64, :, :]), dim=[1, 2, 3])
    a1 = torch.sum(torch.abs(latent[:, 64:, :, :]), dim=[1, 2, 3])
    return (a1 > a0).long()

def final_evaluation(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            recon, latent = model(images, labels)
            loss = forensic_loss(recon, images, latent, labels)
            total_loss += loss.item()
            
            # 여기서 마지막 배치 활성화 강도 기록
            a0 = torch.sum(torch.abs(latent[:, :64, :, :]), dim=[1, 2, 3]).mean()
            a1 = torch.sum(torch.abs(latent[:, 64:, :, :]), dim=[1, 2, 3]).mean()
            print(f"최종 활성화 강도 -> 진짜(a0): {a0:.4f}, 가짜(a1): {a1:.4f}")
            break # 한 배치만 확인
    print(f"최종 총 손실(Average Loss): {total_loss / len(dataloader):.4f}")


# ==========================================
# 4. 학습 루프 (Train Loop)
# ==========================================
# ... (앞부분 생략: 임포트 및 모델 정의는 그대로 유지) ...

if __name__ == '__main__':
    # 모델 초기화
    model = ForensicTransfer().to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 10 # 사전 학습 에포크

    print("🚀 ForensicTransfer 사전 학습(Pre-training)을 시작합니다!")
    
    # 1. 사전 학습 루프 (pretrain_loader 사용)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(pretrain_loader): # 👈 여기를 수정
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            reconstructed, latent = model(images, labels)
            
            loss = forensic_loss(reconstructed, images, latent, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = predict_deepfake(latent)
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            if batch_idx % 10 == 0:
                a0 = torch.sum(torch.abs(latent[:, :64, :, :]), dim=[1, 2, 3]).mean().item()
                a1 = torch.sum(torch.abs(latent[:, 64:, :, :]), dim=[1, 2, 3]).mean().item()
                print(f"  > [Pretrain Batch {batch_idx}] Loss: {loss.item():.4f} | a0: {a0:.2f} | a1: {a1:.2f}")
        
        avg_loss = epoch_loss / len(pretrain_loader) # 👈 여기도 수정
        accuracy = (correct_preds / total_samples) * 100
        
        print(f"📈 Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    print("🎉 사전 학습 완료! 이제 타겟 도메인 성능을 평가합니다.")
    # 가중치 저장
    torch.save(model.state_dict(), 'forensic_model_pretrained.pth')
    print("💾 학습된 모델 가중치가 저장되었습니다.")
    # 2. 저장된 가중치를 불러오기 (이미 메모리에 모델이 있지만, 습관을 들이는 단계입니다)
    model.load_state_dict(torch.load('forensic_model_pretrained.pth', weights_only=True))
    
    # 3. [추가] 타겟 도메인 재학습 (Fine-tuning)
    print("🚀 이제 타겟 데이터로 재학습(Fine-tuning)을 시작합니다!")
    optimizer = optim.Adam(model.parameters(), lr=1e-5) # 학습률을 더 낮게!
    
    model.train()
    for epoch in range(3): 
        for images, labels in target_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            recon, latent = model(images, labels)
            loss = forensic_loss(recon, images, latent, labels)
            loss.backward()
            optimizer.step()
    print("🎉 재학습 완료!")
    # 2. 마지막 평가 루프 (target_loader 사용)
    # 이제 재학습 전 모델이 타겟 데이터를 어떻게 보는지 확인합니다.
    final_evaluation(model, target_loader, device) # 👈 여기를 target_loader로 변경
