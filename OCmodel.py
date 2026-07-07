import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 공통 인코더/디코더 블록 ---
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU()
        )
    def forward(self, x): return self.conv(x).view(x.size(0), -1)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )
    def forward(self, z): return self.deconv(z.view(-1, 128, 8, 8))

# --- 3개 모델 통합 클래스 ---
class OCFakeDectSystem(nn.Module):
    def __init__(self, model_type='OC-FakeDect-2', latent_dim=128):
        super().__init__()
        self.model_type = model_type
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        # VAE/FakeDect용 레이어
        if model_type != 'OC-AE':
            self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
            self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
            self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)
        
        # OC-FakeDect-2 전용 추가 인코더
        if model_type == 'OC-FakeDect-2':
            self.encoder2 = Encoder()
            self.fc_mu2 = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        if self.model_type == 'OC-AE':
            z = self.encoder(x)
            recon = self.decoder(z.view(-1, 128, 8, 8))
            return recon, None, None, None
        
        # VAE 기반 모델 공통
        enc = self.encoder(x)
        mu, logvar = self.fc_mu(enc), self.fc_logvar(enc)
        z = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        recon = self.decoder(self.fc_decode(z).view(-1, 128, 8, 8))
        
        mu2 = None
        if self.model_type == 'OC-FakeDect-2':
            mu2 = self.fc_mu2(self.encoder2(recon))
            
        return recon, mu, logvar, mu2

# --- 논문 수식 기반 손실 함수 ---
def get_loss(recon, x, mu, logvar, model_type):
    mse = F.mse_loss(recon, x, reduction='sum')
    if model_type == 'OC-AE':
        return mse
    
    # KL Divergence (수식 17, 18)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld
