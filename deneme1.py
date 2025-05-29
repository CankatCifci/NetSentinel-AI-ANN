import subprocess
import re
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

# ------------------------------
# Model Sınıfı
# ------------------------------
class BinaryANN(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(BinaryANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ------------------------------
# Tshark Çıktısını Sayısal Özelliklere Çevirme
# ------------------------------
def parse_tshark_output(raw_output):
    lines = raw_output.strip().split('\n')
    processed_data = []
    raw_lines = []

    for line in lines:
        raw_lines.append(line.strip())

        length_match = re.search(r'\s(\d+)$', line.strip())
        length = int(length_match.group(1)) if length_match else 0

        protocol = 0
        if "TCP" in line:
            protocol = 1
        elif "UDP" in line:
            protocol = 2
        elif "SSL" in line or "TLS" in line:
            protocol = 3

        flag = 0
        if "[ACK]" in line:
            flag = 1
        elif "[SYN]" in line:
            flag = 2
        elif "[FIN]" in line:
            flag = 3

        processed_data.append([length, protocol, flag])

    return processed_data, raw_lines

# ------------------------------
# Eğitimli Model Oluşturma
# ------------------------------
def train_and_save_model(path):
    X = []
    y = []
    for _ in range(1000):
        length = np.random.randint(60, 1500)
        protocol = np.random.choice([1, 2, 3])
        flag = np.random.choice([0, 1, 2, 3])
        X.append([length, protocol, flag])
        y.append(1 if length > 1000 and protocol == 3 else 0)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    model = BinaryANN(input_size=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(15):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), path)
    print("✅ Eğitimli model anomaly_detector.pth olarak kaydedildi.")
    return model

# ------------------------------
# Modeli Yükle veya Eğit ve Kaydet
# ------------------------------
model = BinaryANN(input_size=3)
model_path = "anomaly_detector.pth"
today = datetime.now().strftime("%Y-%m-%d")
log_dir = f"logs/{today}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "anomaly_log.txt")
blacklist_file = os.path.join(log_dir, "blacklist.txt")
blacklist_ips = {}

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("✅ Kayıtlı model yüklendi.")
    except:
        print("⚠️ Kayıtlı model yüklenemedi. Yeniden eğitiliyor...")
        model = train_and_save_model(model_path)
else:
    model = train_and_save_model(model_path)

# ------------------------------
# Tshark Komutu
# ------------------------------
tshark_cmd = [
    "C:\\Program Files\\Wireshark\\tshark.exe",
    "-i", "Wi-Fi",
    "-c", "10"
]

# ------------------------------
# Gerçek Zamanlı İzleme Döngüsü
# ------------------------------
print("\n🔍 Gerçek zamanlı trafik analizi başlatıldı...")
while True:
    try:
        result = subprocess.run(tshark_cmd, capture_output=True, text=True)
        raw_output = result.stdout
        processed_data, raw_lines = parse_tshark_output(raw_output)

        if processed_data:
            sample_tensor = torch.tensor(processed_data, dtype=torch.float32)
            with torch.no_grad():
                output = model(sample_tensor)
                predictions = torch.argmax(output, axis=1).numpy()

            anomali_sayisi = 0
            for i, p in enumerate(predictions):
                status = "⚠️ Anormal" if p == 1 else "✅ Normal"
                line = raw_lines[i]
                print(f"Paket {i+1}: {status} | 📄 İçerik: {line}")

                if p == 1:
                    anomali_sayisi += 1
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"ANORMAL PAKET - {time.ctime()}\n{line}\n\n")

                    ip_match = re.search(r"(\d+\.\d+\.\d+\.\d+)", line)
                    if ip_match:
                        ip = ip_match.group(1)
                        blacklist_ips[ip] = blacklist_ips.get(ip, 0) + 1
                        if blacklist_ips[ip] >= 3:
                            with open(blacklist_file, "a", encoding="utf-8") as bl:
                                bl.write(f"{ip} - {blacklist_ips[ip]} anomali\n")

            toplam = len(predictions)
            oran = (anomali_sayisi / toplam) * 100
            print(f"📊 Anormal Oranı: %{oran:.2f}  ({anomali_sayisi}/{toplam})\n")

        time.sleep(2)

    except KeyboardInterrupt:
        print("\n🔒 İzleme sonlandırıldı.")
        break
    except Exception as e:
        print(f"Hata oluştu: {e}")
        time.sleep(5)