import subprocess
import re
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

# Model sınıfı
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

# Tshark çıktısını parse eden fonksiyon (detaylı)
def parse_tshark_output_with_details(raw_output):
    lines = raw_output.strip().split('\n')
    processed_data = []
    raw_lines = []
    packet_details = []  # zaman, kaynak IP, hedef port, ham çıktı

    for line in lines:
        raw_lines.append(line.strip())

        length_match = re.search(r'\s(\d+)$', line.strip())
        length = int(length_match.group(1)) if length_match else 0

        # Protokol tespiti - genişletilmiş
        protocol = 0
        if "TCP" in line:
            if "HTTP" in line:
                protocol = 4  # HTTP
            else:
                protocol = 1  # TCP
        elif "UDP" in line:
            if "DNS" in line:
                protocol = 5  # DNS
            else:
                protocol = 2  # UDP
        elif "SSL" in line or "TLS" in line:
            protocol = 3
        elif "ARP" in line:
            protocol = 6
        else:
            protocol = 0

        flag = 0
        if "[ACK]" in line:
            flag = 1
        elif "[SYN]" in line:
            flag = 2
        elif "[FIN]" in line:
            flag = 3

        time_match = re.match(r'^\s*(\d+\.\d+)', line)
        timestamp = float(time_match.group(1)) if time_match else 0.0

        src_ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+)\s→', line)
        src_ip = src_ip_match.group(1) if src_ip_match else ''

        dst_port_match = re.search(r'→\s(\d+)', line)
        dst_port = int(dst_port_match.group(1)) if dst_port_match else 0

        processed_data.append([length, protocol, flag])
        packet_details.append({
            'timestamp': timestamp,
            'src_ip': src_ip,
            'dst_port': dst_port,
            'raw_line': line.strip()
        })

    return processed_data, raw_lines, packet_details

# Model yükle veya eğit
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
        # Eğitme fonksiyonunu ekleyin
        # model = train_and_save_model(model_path)
else:
    # Eğitme fonksiyonunu ekleyin
    # model = train_and_save_model(model_path)
    pass

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
        processed_data, raw_lines, packet_details = parse_tshark_output_with_details(raw_output)

        if processed_data:
            sample_tensor = torch.tensor(processed_data, dtype=torch.float32)
            with torch.no_grad():
                output = model(sample_tensor)
                probs = torch.softmax(output, dim=1)  # Olasılıkları hesapla
                preds = (probs[:, 1] > 0.65).int().numpy()  # 1. sınıf için 0.5 eşik

            anomali_sayisi = 0
            for i, p in enumerate(preds):
                status = "⚠️ Anormal" if p == 1 else "✅ Normal"
                line = packet_details[i]['raw_line']
                print(f"Paket {i+1}: {status} | 📄 İçerik: {line}")

                if p == 1:
                    anomali_sayisi += 1

                    # Log kaydı
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"⚠️ Saldırı Tespit Edildi!\n")
                        f.write(f"Kaynak IP: {packet_details[i]['src_ip']}\n")
                        f.write(f"Hedef Port: {packet_details[i]['dst_port']}\n")
                        f.write(f"Paket Sayısı: 1\n")
                        f.write(f"Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                    # Kara liste kaydı
                    ip = packet_details[i]['src_ip']
                    if ip:
                        blacklist_ips[ip] = blacklist_ips.get(ip, 0) + 1
                        if blacklist_ips[ip] == 3:
                            with open(blacklist_file, "a", encoding="utf-8") as bl:
                                bl.write(f"{ip} - 3 kere anomali tespit edildi.\n")

            toplam = len(preds)
            oran = (anomali_sayisi / toplam) * 100
            print(f"📊 Anormal Oranı: %{oran:.2f}  ({anomali_sayisi}/{toplam})\n")

        time.sleep(2)

    except KeyboardInterrupt:
        print("\n🔒 İzleme sonlandırıldı.")
        break
    except Exception as e:
        print(f"Hata oluştu: {e}")
        time.sleep(5)
