import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import f1_score

# 1. 모델 아키텍처 정의 (먼저 정의되어야 함)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU()
        )
    def forward(self, x): return self.conv(x).view(x.size(0), -1)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )
    def forward(self, z): return self.deconv(z.view(-1, 128, 8, 8))

class OCFakeDectSystem(nn.Module):
    def __init__(self, model_type, latent_dim=128):
        super().__init__()
        self.model_type = model_type
        self.encoder = Encoder()
        self.decoder = Decoder()
        if model_type != 'OC-AE':
            self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
            self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
            self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)
        if model_type == 'OC-FakeDect-2':
            self.encoder2 = Encoder()
            self.fc_mu2 = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        if self.model_type == 'OC-AE':
            z = self.encoder(x)
            recon = self.decoder(z.view(-1, 128, 8, 8))
            return recon, None, None, None
        enc = self.encoder(x)
        mu, logvar = self.fc_mu(enc), self.fc_logvar(enc)
        z = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        recon = self.decoder(self.fc_decode(z).view(-1, 128, 8, 8))
        mu2 = self.fc_mu2(self.encoder2(recon)) if self.model_type == 'OC-FakeDect-2' else None
        return recon, mu, logvar, mu2

# 2. 실험 실행 함수 정의
def run_experiment(model_type, train_loader, test_loader, device):
    model = OCFakeDectSystem(model_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 학습 (Real만 사용)
    model.train()
    for epoch in range(5):
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar, _ = model(x)
            loss = nn.functional.mse_loss(recon, x, reduction='sum')
            if mu is not None:
                loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss.backward()
            optimizer.step()
            
    # 평가
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            recon, mu, _, mu2 = model(x)
            if model_type == 'OC-FakeDect-2':
                score = torch.sqrt(torch.mean((mu - mu2)**2, dim=1))
            else:
                score = torch.sqrt(torch.mean((recon - x)**2, dim=[1,2,3]))
            scores.extend(score.cpu().numpy()); labels.extend(y.numpy())
            
    threshold = np.percentile(scores, 90)
    return f1_score(labels, (np.array(scores) > threshold).astype(int))

# 3. 메인 실행부
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform_data = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    train_loader = DataLoader(datasets.ImageFolder('C:/Users/배나현/Desktop/code실습/Train_OC', transform=transform_data), batch_size=32, shuffle=True)
    test_loader = DataLoader(datasets.ImageFolder('C:/Users/배나현/Desktop/code실습/Test_OC', transform=transform_data), batch_size=32, shuffle=False)

    for m_type in ['OC-AE', 'OC-FakeDect-1', 'OC-FakeDect-2']:
        f1 = run_experiment(m_type, train_loader, test_loader, device)
        print(f"[{m_type}] 최종 F1-Score: {f1:.4f}")
