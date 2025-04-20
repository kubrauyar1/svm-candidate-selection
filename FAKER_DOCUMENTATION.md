# Faker Kütüphanesi Dokümantasyonu

## Faker Nedir?

Faker, Python için geliştirilmiş bir sahte veri üretme kütüphanesidir. Gerçekçi görünen ancak tamamen rastgele üretilen veriler oluşturmak için kullanılır. Bu özelliği ile test verileri oluşturma, demo uygulamalar geliştirme ve veri gizleme gibi birçok alanda kullanılabilir.

## Temel Özellikler

1. **Çoklu Dil Desteği**: 20'den fazla dilde veri üretebilir
2. **Zengin Veri Tipleri**: İsim, adres, telefon, e-posta, şirket bilgileri vb.
3. **Özelleştirilebilir**: Kendi sağlayıcılarınızı ekleyebilirsiniz
4. **Yerelleştirilmiş Veri**: Her dil için o dile özgü veri formatları

## Kurulum

```bash
pip install Faker
```

## Temel Kullanım

```python
from faker import Faker

# Faker nesnesi oluştur
fake = Faker()

# Türkçe veri üretmek için
fake_tr = Faker('tr_TR')

# Temel veri üretme örnekleri
print(fake.name())          # Rastgele isim
print(fake.address())       # Rastgele adres
print(fake.email())         # Rastgele e-posta
print(fake.company())       # Rastgele şirket adı
print(fake.job())           # Rastgele meslek
print(fake.phone_number())  # Rastgele telefon numarası
```

## İşe Alım Projesi için Örnek Kullanım

```python
from faker import Faker
import numpy as np

fake = Faker('tr_TR')

def generate_applicant_data(n_samples):
    data = []
    for _ in range(n_samples):
        applicant = {
            'ad_soyad': fake.name(),
            'tecrube_yili': np.random.uniform(0, 10),
            'teknik_puan': np.random.uniform(0, 100),
            'universite': fake.company() + ' Üniversitesi',
            'bolum': fake.job() + ' Mühendisliği',
            'email': fake.email(),
            'telefon': fake.phone_number()
        }
        data.append(applicant)
    return data

# 10 aday için veri üret
applicants = generate_applicant_data(10)
for applicant in applicants:
    print(applicant)
```

## Özel Sağlayıcı Oluşturma

```python
from faker import Faker
from faker.providers import BaseProvider

class CustomProvider(BaseProvider):
    def programming_language(self):
        languages = ['Python', 'Java', 'JavaScript', 'C++', 'C#', 'Ruby', 'Go']
        return self.random_element(languages)

fake = Faker()
fake.add_provider(CustomProvider)

print(fake.programming_language())  # Rastgele programlama dili
```

## Veri Seti Oluşturma ve CSV'ye Kaydetme

```python
import pandas as pd
from faker import Faker

fake = Faker('tr_TR')

def generate_dataset(n_samples):
    data = []
    for _ in range(n_samples):
        row = {
            'ad_soyad': fake.name(),
            'tecrube_yili': np.random.uniform(0, 10),
            'teknik_puan': np.random.uniform(0, 100),
            'universite': fake.company() + ' Üniversitesi',
            'bolum': fake.job() + ' Mühendisliği',
            'email': fake.email(),
            'telefon': fake.phone_number(),
            'programlama_dili': fake.programming_language()
        }
        data.append(row)
    return pd.DataFrame(data)

# 100 aday için veri seti oluştur
df = generate_dataset(100)

# CSV'ye kaydet
df.to_csv('aday_verileri.csv', index=False)
```

## Faker'ın Avantajları

1. **Gerçekçi Veri**: Üretilen veriler gerçek dünya verilerine benzer
2. **Hızlı Veri Üretimi**: Büyük veri setleri hızlıca oluşturulabilir
3. **Test Kolaylığı**: Test senaryoları için ideal
4. **Gizlilik**: Gerçek veriler yerine sahte veriler kullanılabilir
5. **Özelleştirilebilirlik**: İhtiyaca göre özelleştirilebilir

## Sınırlamalar

1. Veriler tamamen rastgele olduğu için gerçek veri setlerindeki korelasyonları yakalayamayabilir
2. Bazı durumlarda veri tutarlılığı sağlamak zor olabilir
3. Çok spesifik veri tipleri için özel sağlayıcılar gerekebilir

## İpuçları

1. Veri üretirken seed değeri kullanarak tekrarlanabilirlik sağlayın
2. Özel ihtiyaçlarınız için kendi sağlayıcılarınızı oluşturun
3. Büyük veri setleri için generator pattern kullanın
4. Veri tutarlılığı için ilişkili verileri birlikte üretin 