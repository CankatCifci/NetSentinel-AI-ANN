import subprocess
import re
import time
import torch
import torch.nn as nn
import numpy as np
import json
import os
from datetime import datetime

# ------------------------------
# Model Sınıfı
# ------------------------------
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

# ------------------------------
# Model ve Özellikleri Yükle
# ------------------------------
with open("label_map.json", "r") as f:
    label_map = json.load(f)

with open("input_feature_names.json", "r") as f:
    expected_features = json.load(f)

model_data = torch.load("anomaly_detector.pth")
model = Net(model_data['input_size'], 128, model_data['output_size'])
model.load_state_dict(model_data['model_state_dict'])
model.eval()

# ------------------------------
# Log klasörü
# ------------------------------
today = datetime.now().strftime("%Y-%m-%d")
log_dir = f"logs/{today}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "anomaly_log.txt")
blacklist_file = os.path.join(log_dir, "blacklist.txt")
blacklist_ips = {}

# ------------------------------
# Tshark komutu (örnek)
# ------------------------------
tshark_cmd = [
    "C:\\Program Files\\Wireshark\\tshark.exe",
    "-i", "Wi-Fi",
    "-c", "10"
]

# ------------------------------
# Öznitelikleri sahte üret (demo amaçlı)
# ------------------------------
def simulate_features():
    # Gerçek tshark entegrasyonu yerine demo amaçlı rastgele veri üretiliyor
    features = []
    for _ in range(10):
        row = np.random.rand(len(expected_features))  # tüm özellikler için değer
        features.append(row)
    return torch.tensor(features, dtype=torch.float32)

# ------------------------------
# Gerçek zamanlı döngü
# ------------------------------
print("\n🔍 Gerçek zamanlı trafik analizi başlatıldı...")
while True:
    try:
        X_tensor = simulate_features()
        with torch.no_grad():
            outputs = model(X_tensor)
            preds = torch.argmax(outputs, dim=1).numpy()

        for i, pred in enumerate(preds):
            label = label_map[str(pred)]
            status = "✅" if label.lower() == "benign" else "⚠️"
            print(f"Paket {i+1}: {status} {label}")

        time.sleep(2)

    except KeyboardInterrupt:
        print("\n🔒 İzleme sonlandırıldı.")
        break
    except Exception as e:
        print(f"Hata oluştu: {e}")
        time.sleep(5)