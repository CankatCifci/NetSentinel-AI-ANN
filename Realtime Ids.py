import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
import os

# Veri setini oku (hatalı satırları atla)
df = pd.read_csv("Darknet.CSV", low_memory=False, on_bad_lines='skip')

# Gereksiz sütunları çıkar
ignore_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
df = df.drop(columns=[col for col in ignore_cols if col in df.columns], errors='ignore')

# NaN ve sonsuz değerleri kaldır
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Özellik ve etiket ayır
y = df['Label']
X = df.drop(columns=['Label', 'Label.1'], errors='ignore')

# Özellik isimlerini kaydet
with open("input_feature_names.json", "w") as f:
    json.dump(X.columns.tolist(), f)

# Label encoding
y_encoded = LabelEncoder().fit_transform(y)

# Sınıf haritasını kaydet
label_map = {i: label for i, label in enumerate(np.unique(y))}
with open("label_map.json", "w") as f:
    json.dump(label_map, f)

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim/test böl
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# PyTorch tensörleri
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Model tanımı
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Model oluştur
input_size = X.shape[1]
hidden_size = 128
output_size = len(np.unique(y_encoded))

model = Net(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim
epochs = 30
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# Modeli kaydet
torch.save({
    "model_state_dict": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size
}, "anomaly_detector.pth")

print("\n✅ Geniş veri modeli anomaly_detector.pth, label_map.json ve input_feature_names.json başarıyla kaydedildi.")
