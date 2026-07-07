import matplotlib.pyplot as plt
import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('training_log.csv', names=['Epoch', 'Loss', 'Accuracy'])

plt.figure(figsize=(12, 5))

# 1. Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(df['Epoch'], df['Loss'], marker='o', color='red')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# 2. Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(df['Epoch'], df['Accuracy'], marker='o', color='blue')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.savefig('result_plot.png') # 이미지 파일로 저장
plt.show()
