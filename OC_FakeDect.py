import torch
import torch_directml
import torch.nn as nn
import torch.nn.functional as F

device = torch_directml.device()
print(f"현재 사용 중인 연산 장치: {device}")

# -----------------------------------------------------
# 1. OC-FakeDect-2 모델 아키텍처 (논문 그림 14-b 반영)
# -----------------------------------------------------
class OCFakeDect(nn.Module):
    def __init__(self, latent_dim=128):
        super(OCFakeDect, self).__init__()
        
        # [첫 번째 인코더] - 원본 이미지용
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc_mu1 = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar1 = nn.Linear(128 * 8 * 8, latent_dim)
        
        # [디코더] - 잠재 벡터 Z로부터 이미지 재구성
        self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() 
        )

        # ★ 논문 핵심: [두 번째 인코더] - 복원된 이미지의 특징 추출용
        # (구조는 첫 번째 인코더와 동일하게 구성)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc_mu2 = nn.Linear(128 * 8 * 8, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 1. 원본 이미지 인코딩
        enc1_out = self.encoder1(x)
        enc1_out = enc1_out.view(enc1_out.size(0), -1)
        mu1 = self.fc_mu1(enc1_out)
        logvar1 = self.fc_logvar1(enc1_out)
        
        # 2. 재매개변수화 및 이미지 복원
        z = self.reparameterize(mu1, logvar1)
        dec_in = self.fc_decode(z)
        dec_in = dec_in.view(dec_in.size(0), 128, 8, 8)
        reconstructed_x = self.decoder(dec_in)
        
        # 3. ★ 복원된 이미지를 두 번째 인코더에 통과시켜 새로운 mu 추출
        enc2_out = self.encoder2(reconstructed_x)
        enc2_out = enc2_out.view(enc2_out.size(0), -1)
        mu2 = self.fc_mu2(enc2_out)
        
        # 반환값에 mu2 추가
        return reconstructed_x, mu1, logvar1, mu2

# -----------------------------------------------------
# 2. 손실 함수 및 이상 점수 계산
# -----------------------------------------------------
def oc_fakedect_loss(reconstructed_x, x, mu1, logvar1):
    # Loss는 기존과 동일하게 원본 픽셀 복원력과 KL Divergence로 학습합니다.
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
    return recon_loss + kl_divergence

def calculate_anomaly_score_rmse(mu1, mu2):
    """
    ★ 논문 3.3.3절 반영: OC-FakeDect-2의 이상 점수 계산
    픽셀(X)이 아니라 잠재 공간의 특징인 mu1(X)와 mu2(X') 사이의 RMSE를 구합니다.
    """
    # 잠재 공간 차원(latent_dim)을 기준으로 오차를 구함
    mse = F.mse_loss(mu1, mu2, reduction='none')
    mse_per_latent = torch.mean(mse, dim=1) 
    
    rmse_score = torch.sqrt(mse_per_latent)
    return rmse_score
