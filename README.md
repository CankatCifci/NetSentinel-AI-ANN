![image](https://github.com/user-attachments/assets/4b35a0d1-4871-4b12-acc6-204ebd61e4db)Bu proje, gerçek zamanlı ağ trafiğini analiz ederek anomali tespiti yapmayı amaçlayan bir yapay zeka destekli siber güvenlik sistemidir.
Wireshark'ın komut satırı aracı olan tshark ile veriler toplanır, işlenir ve daha önceden eğitilmiş bir yapay sinir ağı modeliyle olası saldırılar tespit edilir.

Proje Dosya Yapısı
📦 anomaly-detector/
 ┣ 📁 logs/             # Anomali loglarının ve kara liste kayıtlarının tutulduğu klasör
 ┣ 📄 anomaly_detector.pth        # Eğitilmiş model dosyası
 ┣ 📄 label_map.json              # Sınıf isimleri haritası
 ┣ 📄 Darknet.CSV                 # Kullanılan veri seti
 ┣ 📄 ModelTraining.py            # Model eğitim scripti
 ┣ 📄 deneme 4 zamanlı.py         # Gerçek zamanlı izleme scripti

 Python 3.8+ ile çalışır.
 pip install torch pandas numpy scikit-learn
 Wireshark'ı yükledikten sonra, tshark.exe yolunu  deneme 4 zamanlı.py içinde kendi sistemine göre güncelle:

 tshark_cmd = [
    "C:\\Program Files\\Wireshark\\tshark.exe",  # Windows için yol
    "-i", "Wi-Fi",  # Ağ arayüzü ismini kendi sistemine göre değiştir
    "-c", "10"
]

Proje, Darknet Traffic Dataset adlı kamuya açık bir siber saldırı veri setini kullanır.
Darknet.CSV

Model, bu veri seti üzerinde çok sınıflı olarak eğitilir. Eğitimin sonunda sadece normal ve anormal olarak (binary) ayrım yapan versiyonu anomaly_detector.pth olarak kaydedilir.

Eğitim Parametreleri:
Gizli katman: 128 nöron
Aktivasyon: ReLU
Epoch: 30
Optimizasyon: Adam
Kayıp Fonksiyonu: CrossEntropyLoss
Çıktılar:
anomaly_detector.pth – Eğitilmiş model ağırlıkları
label_map.json – Sınıf ID – isim eşlemesi

 Gerçek Zamanlı Anomali Tespiti ( deneme 4 zamanlı.py)
Bu script, gerçek zamanlı paket verilerini tshark ile alır ve her paketi eğitilmiş modele göndererek “normal” ya da “anormal” sınıflandırması yapar.
Kullanılan Özellikler:
Paket uzunluğu
Protokol türü (TCP, UDP, HTTP, TLS, DNS, ARP)
Bayrak bilgisi ([SYN], [ACK], [FIN])

Kayıtlar:
logs/YYYY-MM-DD/anomaly_log.txt – Anormal paket logları
logs/YYYY-MM-DD/blacklist.txt – 3 defa anomali yapan IP'ler

Kullanım Senaryoları
Üniversite ve şirket ağlarında saldırı tespiti
SOC (Security Operation Center) sistemleri için temel modül
Eğitim amaçlı anomali tespiti örneği
