import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ 현재 사용 중인 연산 장치: {device}")

# ==========================================
# 0. 데이터셋 경로 설정 (아까 만든 폴더 그대로 사용!)
# ==========================================
SOURCE_DATA_DIR = './Source_Dataset' # 학습용 (여기서 Real만 빼서 쓸 예정)
TARGET_DATA_DIR = './Target_Dataset' # 평가용 (T-GD와 동일한 타겟 데이터)

# ==========================================
# 1. OC-FakeDect-2 모델 정의 (논문 완벽 반영)
# ==========================================
class OCFakeDect(nn.Module):
    def __init__(self, latent_dim=128):
        super(OCFakeDect, self).__init__()
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU()
        )
        self.fc_mu1 = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar1 = nn.Linear(128 * 8 * 8, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), nn.Sigmoid() 
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU()
        )
        self.fc_mu2 = nn.Linear(128 * 8 * 8, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc1_out = self.encoder1(x).view(x.size(0), -1)
        mu1 = self.fc_mu1(enc1_out)
        logvar1 = self.fc_logvar1(enc1_out)
        
        z = self.reparameterize(mu1, logvar1)
        dec_in = self.fc_decode(z).view(z.size(0), 128, 8, 8)
        reconstructed_x = self.decoder(dec_in)
        
        enc2_out = self.encoder2(reconstructed_x).view(reconstructed_x.size(0), -1)
        mu2 = self.fc_mu2(enc2_out)
        
        return reconstructed_x, mu1, logvar1, mu2

# ==========================================
# 2. 손실 함수 (학습용) 및 이상 점수 계산 (평가용)
# ==========================================
def oc_fakedect_loss(reconstructed_x, x, mu1, logvar1):
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
    return recon_loss + kl_divergence

def calculate_anomaly_score(mu1, mu2):
    # 잠재 공간의 RMSE 계산 (값이 높을수록 가짜 이미지)
    mse = F.mse_loss(mu1, mu2, reduction='none')
    return torch.sqrt(torch.mean(mse, dim=1))

if __name__ == '__main__':
    # 64x64 사이즈 권장 (CNN 레이어 계산 용이)
    transform = transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.ToTensor(),
    ])
    
    # 데이터 로더 세팅
    train_dataset = datasets.ImageFolder(root=SOURCE_DATA_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(root=TARGET_DATA_DIR, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 클래스 번호 확인 (아까 로그에서 training_real이 1이었습니다)
    REAL_LABEL_IDX = train_dataset.class_to_idx['training_real']
    FAKE_LABEL_IDX = train_dataset.class_to_idx['training_fake']
    
    model = OCFakeDect().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # ==========================================
    # [1단계] One-Class 학습 (진짜 이미지만 학습!)
    # ==========================================
    print("🚀 [1단계] 진짜(Real) 얼굴만 사용하여 OC-FakeDect-2 학습을 시작합니다!")
    epochs = 5 # VAE는 CNN보다 학습이 빠르므로 5 에포크 진행
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            # ★ 핵심: 진짜(Real) 이미지만 필터링해서 학습!
            real_inputs = inputs[labels == REAL_LABEL_IDX].to(device)
            
            if len(real_inputs) == 0:
                continue # 배치 안에 진짜 이미지가 없으면 패스
            
            optimizer.zero_grad()
            reconstructed_x, mu1, logvar1, _ = model(real_inputs)
            
            loss = oc_fakedect_loss(reconstructed_x, real_inputs, mu1, logvar1)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"📈 Epoch [{epoch+1}/{epochs}] 완료 | 모델이 정상 얼굴의 분포를 외우고 있습니다.")

    # ==========================================
    # [2단계] Target 도메인 평가 (T-GD와 성능 비교)
    # ==========================================
    print("\n📊 [2단계] 열화된 Target 데이터를 투입하여 이상치(Fake)를 솎아냅니다.")
    model.eval()
    
    all_true_labels = []
    all_anomaly_scores = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            _, mu1, _, mu2 = model(inputs)
            
            # 이상 점수(RMSE) 계산
            anomaly_scores = calculate_anomaly_score(mu1, mu2)
            
            # AUROC 계산을 위해 정답 라벨 변환 (Fake면 1, Real이면 0으로 맞춰줌)
            binary_labels = (labels == FAKE_LABEL_IDX).float() 
            
            all_true_labels.extend(binary_labels.cpu().numpy())
            all_anomaly_scores.extend(anomaly_scores.cpu().numpy())

    # AUROC 계산
    auroc = roc_auc_score(all_true_labels, all_anomaly_scores)
    
    print("==================================================")
    print(f"🏆 최종 Target 도메인 AUROC 점수 (OC-FakeDect): {auroc * 100:.2f}%")
    print(f"💡 (비교용) 아까 T-GD의 최고 점수: 57.92%")
    print("==================================================")
