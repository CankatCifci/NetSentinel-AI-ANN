import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Veri setini oku
df = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv")

# Sütun isimlerindeki boşlukları temizle
df.columns = df.columns.str.strip()

print(df.columns.tolist())  # Kontrol amaçlı

X = df.drop(columns=['Label'])
y = df['Label']

# Sonsuz değerleri NaN yap
X = X.replace([np.inf, -np.inf], np.nan)

# NaN içeren satırları düşür
X = X.dropna(axis=0)
y = y.loc[X.index]  # Etiketleri de eşleştir

# Label encoding (etiketleri sayıya çevir)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Özellikleri standartlaştır (ortalama 0, std 1 olacak şekilde)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test olarak böl (örnek olarak %20 test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# PyTorch tensörlerine çevir
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Model sınıfı
class BinaryANN(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(BinaryANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, len(label_encoder.classes_))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Modeli oluştur
input_size = X_train.shape[1]
model = BinaryANN(input_size=input_size)

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim döngüsü
epochs = 30
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# Model kaydetme
torch.save(model.state_dict(), "anomaly_detector.pth")
print("✅ Model başarıyla kaydedildi.")

# Test performans metrikleri
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, test_preds = torch.max(test_outputs, 1)

y_true = y_test.numpy()
y_pred = test_preds.numpy()

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
accuracy = accuracy_score(y_true, y_pred)

print("\nModel Test Performans Metrikleri:")
print(f"Doğruluk (Accuracy): {accuracy:.4f}")
print(f"Kesinlik (Precision): {precision:.4f}")
print(f"Duyarlılık (Recall): {recall:.4f}")
print(f"F1 Skoru: {f1:.4f}")
