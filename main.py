import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import subprocess
import re

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

# Confusion Matrix hesaplama
cm = confusion_matrix(y_test, y_pred)
print("\n🧮 Confusion Matrix:")
print(cm)

# Detection Rate (Doğruluk Oranı) hesaplama
TP = cm[1, 1]
FN = cm[1, 0]
Detection_Rate = TP / (TP + FN)
print(f"Detection Rate (Doğruluk Oranı): {Detection_Rate:.4f}")

# False Alarm Rate (Yanlış Alarm Oranı) hesaplama
FP = cm[0, 1]
TN = cm[0, 0]
False_Alarm_Rate = FP / (FP + TN)
print(f"False Alarm Rate (Yanlış Alarm Oranı): {False_Alarm_Rate:.4f}")

# Tshark komutunu çalıştırmak için subprocess kullanıyoruz
tshark_path = "C:\\Program Files\\Wireshark\\tshark.exe"
interface = 'Wi-Fi'  # Ya da 'Ethernet'

# Tshark komutunu çalıştırıyoruz
tshark_command = [tshark_path, "-i", interface, "-c", "10"]  # Burada "-c 10" ile 10 paket yakalıyoruz


# Tshark çıktısını işle
def process_tshark_output(output):
    """
    Tshark çıktısını işle ve modelin anlayacağı formata dönüştür.
    Bu fonksiyon, tshark çıktısından gerekli ağ trafiği verilerini çıkarır
    ve bunları modelin tahmin yapabileceği sayısal verilere dönüştürür.
    """
    processed_data = []

    for line in output.splitlines():
        # Zaman damgası (ilk sütun)
        time_match = re.match(r"^\s*(\d+\.\d+)\s", line)
        if time_match:
            timestamp = float(time_match.group(1))
        else:
            continue

        # Kaynak ve hedef IP'ler (IP adresi arayışı)
        ip_match = re.search(r"(\d+\.\d+\.\d+\.\d+)\s→\s(\d+\.\d+\.\d+\.\d+)", line)
        if ip_match:
            src_ip = ip_match.group(1)
            dst_ip = ip_match.group(2)
        else:
            src_ip = dst_ip = None

        # Protokol türü (TLS, TCP, DNS, vs.)
        protocol_match = re.search(r"\s([A-Za-z0-9]+)\s", line)
        if protocol_match:
            protocol = protocol_match.group(1)
        else:
            protocol = None

        # Port numaraları (Kaynak ve hedef portları)
        port_match = re.search(r"(\d+)\s→\s(\d+)", line)
        if port_match:
            src_port = int(port_match.group(1))
            dst_port = int(port_match.group(2))
        else:
            src_port = dst_port = None

        # Paket uzunluğu (Len=xxx)
        length_match = re.search(r"Len=(\d+)", line)
        if length_match:
            length = int(length_match.group(1))
        else:
            length = None

        # Bayraklar (ACK, SYN, FIN vb.)
        flags = None
        if "[ACK]" in line:
            flags = "ACK"
        elif "[SYN]" in line:
            flags = "SYN"
        elif "[FIN]" in line:
            flags = "FIN"

        # Çıkarılan veriyi sayısal formata dönüştürme
        processed_data.append([
            timestamp,  # Zaman damgası
            src_ip,  # Kaynak IP
            dst_ip,  # Hedef IP
            protocol,  # Protokol
            src_port,  # Kaynak port
            dst_port,  # Hedef port
            length,  # Paket uzunluğu
            flags  # Bayraklar
        ])

    processed_data = np.array(processed_data)
    return processed_data


# Gerçek zamanlı izleme sırasında tshark çıktısını işle
try:
    while True:
        result = subprocess.run(tshark_command, capture_output=True, text=True, check=True, encoding='utf-8')
        print(result.stdout)

        # Tshark çıktısını işle
        processed_data = process_tshark_output(result.stdout)

        # Model ile tahmin yap
        if processed_data.size > 0:  # Eğer işlenmiş veri varsa
            processed_tensor = torch.tensor(processed_data, dtype=torch.float32)
            with torch.no_grad():
                model.eval()
                output = model(processed_tensor)
                predicted_class = torch.argmax(output, axis=1).item()

            # Sonucu yorumla
            if predicted_class == 0:  # Örneğin, 0 'güvenli' kategorisini temsil ediyor
                print("Güvenli")
            else:
                print("Anormal: Güvenli Değil")

except subprocess.CalledProcessError as e:
    print(f"Tshark komutunu çalıştırırken hata oluştu: {e}")
except UnicodeDecodeError as e:
    print("Hata: Tshark çıktısı unicode hatası veriyor. Kodlama hatası çözümü gerekiyor.")
    result = subprocess.run(tshark_command, capture_output=True, text=True, check=True, encoding='latin-1')
    print(result.stdout)
