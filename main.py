import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Veri setini yükle
df = pd.read_csv("Darknet.CSV", on_bad_lines='skip')



# Gereksiz sütunları kaldır
drop_columns = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
df = df.drop(columns=drop_columns, errors='ignore')

# Eksik verileri temizle
df = df.replace([np.inf, -np.inf], np.nan)  # Sonsuz değerleri kaldır
df = df.dropna()

# Hedef değişkeni belirle
y = df['Label']  # veya 'Label.1'
X = df.drop(columns=['Label', 'Label.1'], errors='ignore')

# Kategorik verileri sayısallaştır (One-Hot Encoding)
X = pd.get_dummies(X)

# Hedef değişkeni encode et
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Sınıf isimlerini sayıya çevir (0, 1, 2, 3)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Veriyi PyTorch tensörlerine çevir
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# Yapay Sinir Ağı Modeli Tanımla
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# Model parametreleri
input_size = X_train.shape[1]
hidden_size = 128
output_size = len(np.unique(y))

# Modeli oluştur
model = ANN(input_size, hidden_size, output_size)

# Kayıp fonksiyonu ve optimizasyon yöntemi
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Model eğitimi
epochs = 50
batch_size = 64
for epoch in range(epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i + batch_size]
        y_batch = y_train_tensor[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Test seti ile model değerlendirme
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = torch.argmax(y_pred_tensor, axis=1).numpy()

# Doğruluk oranı ve sınıflandırma raporu
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluk Oranı: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
