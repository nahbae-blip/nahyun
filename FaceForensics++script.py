import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, test_loader, device, model_type='OC-FakeDect-2'):
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            
            # 모델에 따라 추론 방식이 다름
            if model_type == 'OC-AE':
                recon, _, _, _ = model(inputs)
                scores = torch.sqrt(torch.mean((recon - inputs)**2, dim=[1,2,3]))
                
            elif model_type == 'OC-FakeDect-1':
                recon, _, _, _ = model(inputs)
                scores = torch.sqrt(torch.mean((recon - inputs)**2, dim=[1,2,3]))
                
            elif model_type == 'OC-FakeDect-2':
                _, mu1, _, mu2 = model(inputs)
                scores = torch.sqrt(torch.mean((mu1 - mu2)**2, dim=1))

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy()) # 0: Real, 1: Fake

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # 논문 기준 평가: 이상 점수 상위 10%를 Fake(1)로 판정 (실험 환경에 따라 조절)
    threshold = np.percentile(all_scores, 90)
    predictions = (all_scores > threshold).astype(int)

    # 지표 계산
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)

    return precision, recall, f1

# --- 사용 예시 ---
# models = ['OC-AE', 'OC-FakeDect-1', 'OC-FakeDect-2']
# for m_type in models:
#     model = ModelFactory.get_model(m_type).to(device)
#     p, r, f1 = evaluate_model(model, test_loader, device, m_type)
#     print(f"[{m_type}] Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
