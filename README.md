# İşe Alımda Aday Seçimi: SVM ile Başvuru Değerlendirme

Bu proje, yazılım geliştirici pozisyonu için başvuran adayların tecrübe yılı ve teknik sınav puanına göre işe alınıp alınmamasını tahmin eden bir makine öğrenmesi modeli içerir.

## Proje Yapısı

```
.
├── src/                    # Kaynak kodlar
│   ├── __init__.py        # Paket tanımı
│   ├── main.py            # Ana çalıştırma dosyası
│   ├── models/            # Model sınıfları
│   │   └── svm_model.py   # SVM modeli
│   ├── api/               # API kodları
│   │   └── app.py         # FastAPI uygulaması
│   └── utils/             # Yardımcı fonksiyonlar
├── data/                  # Veri klasörü
│   ├── raw/              # Ham veriler
│   └── processed/        # İşlenmiş veriler
├── docs/                  # Dokümantasyon
├── tests/                 # Test dosyaları
├── plots/                 # Görseller
├── requirements.txt       # Bağımlılıklar
└── README.md             # Proje dokümantasyonu
```

## Özellikler

- SVM (Support Vector Machine) kullanarak aday değerlendirme
- Farklı kernel fonksiyonları ile deneyim (linear, rbf, poly, sigmoid)
- Hyperparameter tuning (C ve gamma parametreleri)
- FastAPI ile REST API servisi
- Görselleştirme ve karar sınırı analizi
- Otomatik görsel kaydetme ve raporlama

## Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/kullanici_adi/proje_adi.git
cd proje_adi
```

2. Sanal ortam oluşturun ve aktifleştirin:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
.\venv\Scripts\activate  # Windows
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

### Model Eğitimi ve Değerlendirme

Ana modeli çalıştırmak için:
```bash
python src/main.py
```

Bu komut:
- Sentetik veri oluşturur
- Modeli eğitir
- Farklı kernel fonksiyonlarını dener
- Hyperparameter tuning yapar
- Karar sınırını görselleştirir
- Kullanıcıdan girdi alarak tahmin yapar
- Tüm görselleri `plots/` klasörüne kaydeder

### API Servisi

API servisini başlatmak için:
```bash
uvicorn src.api.app:app --reload
```

API'yi test etmek için:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"tecrube_yili": 3, "teknik_puan": 75}'
```

## Model Kriterleri

Adaylar aşağıdaki kriterlere göre değerlendirilir:
- Tecrübe < 2 yıl VE teknik puan < 60: İşe alınmaz (1)
- Diğer durumlar: İşe alınır (0)

## Görseller

Model eğitimi sırasında oluşturulan görseller `plots/` klasöründe saklanır:
- `linear_kernel_decision_boundary.png`: Linear kernel için karar sınırı
- `rbf_parameter_performance.png`: RBF kernel için parametre performansı
- `poly_parameter_performance.png`: Polynomial kernel için parametre performansı
- `sigmoid_parameter_performance.png`: Sigmoid kernel için parametre performansı
- `best_model_decision_boundary.png`: En iyi model için karar sınırı

## API Endpoints

- `GET /`: API hakkında bilgi
- `POST /predict`: Aday değerlendirmesi yapar

## Gereksinimler

- Python 3.8+
- numpy
- matplotlib
- scikit-learn
- fastapi
- uvicorn

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın. 