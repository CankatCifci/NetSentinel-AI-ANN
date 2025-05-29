import subprocess
import re
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

# Model sınıfı - eğitimde kullandığınızla aynı olmalı
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

# Tshark çıktısını parse eden fonksiyon (örnek - kullandığınız özelliklere göre uyarlayın)
def parse_tshark_output_with_details(raw_output):
    lines = raw_output.strip().split('\n')
    processed_data = []
    raw_lines = []
    for line in lines:
        raw_lines.append(line.strip())

        # Buraya eğitimde kullandığınız 3 özellikten örnek:
        # Paket uzunluğu
        length_match = re.search(r'\s(\d+)$', line.strip())
        length = int(length_match.group(1)) if length_match else 0

        # Protokol (TCP=1, UDP=2, SSL=3)
        protocol = 0
        if "TCP" in line:
            protocol = 1
        elif "UDP" in line:
            protocol = 2
        elif "SSL" in line or "TLS" in line:
            protocol = 3

        # Bayrak (ACK=1, SYN=2, FIN=3)
        flag = 0
        if "[ACK]" in line:
            flag = 1
        elif "[SYN]" in line:
            flag = 2
        elif "[FIN]" in line:
            flag = 3

        processed_data.append([length, protocol, flag])

    return processed_data, raw_lines

# Modeli yükle
model = BinaryANN(input_size=3)
model_path = "anomaly_detector.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("✅ Model başarıyla yüklendi.")
else:
    print("❌ Model dosyası bulunamadı!")

# Tshark komutu
tshark_cmd = [
    "C:\\Program Files\\Wireshark\\tshark.exe",
    "-i", "Wi-Fi",
    "-c", "10"
]

print("\n🔍 Gerçek zamanlı trafik analizi başlatıldı...")
while True:
    try:
        result = subprocess.run(tshark_cmd, capture_output=True, text=True)
        raw_output = result.stdout

        processed_data, raw_lines = parse_tshark_output_with_details(raw_output)
        if not processed_data:
            time.sleep(2)
            continue

        # Torch tensöre çevir
        sample_tensor = torch.tensor(processed_data, dtype=torch.float32)

        # Tahmin
        with torch.no_grad():
            output = model(sample_tensor)
            predictions = torch.argmax(output, axis=1).numpy()

        # Sonuçları yazdır
        for i, p in enumerate(predictions):
            status = "⚠️ Anormal" if p == 1 else "✅ Normal"
            print(f"Paket {i+1}: {status} | İçerik: {raw_lines[i]}")

        time.sleep(2)

    except KeyboardInterrupt:
        print("\n🔒 İzleme sonlandırıldı.")
        break
    except Exception as e:
        print(f"Hata oluştu: {e}")
        time.sleep(5)
