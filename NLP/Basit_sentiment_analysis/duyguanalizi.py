"""
Amaç: Duygu analizi,
Duygu analizi yapmak için TF-IDF ile Logistic Regression sınıflandırıcı kullanılmıştır
Birden fazla kategori olduğu için sigmoid değil softmax fonksiyonu kullanılmıştır
Veri seti: Farklı platformlardan etiketli cümleler, yorumlar içeren bir veri seti kullanılmıştır
yorumların pozitif, negatif veya nötr olarak etiketlendiği bir veri setidir

pip install pandas scikit-learn nltk datasets

Adımlar

1. Kütüphanelerin Kurulumu ve dahil edilemsi.
2. Veri Yükleme ve Temizlik: CSV dosyasının okunması ve eksik/boş verilerin (NaN) veri setinden çıkarılması.
3. Pre-processing
4. Veri Bölme (Train/Test Split)
5. Vektörizasyon (TF-IDF)
6. Model Eğitimi: Lojistik Regresyon algoritmasının oluşturulması ve eğitim (train) verisiyle modelin eğitilmesi.
7. Performans Ölçümü, Accuracy skoru
8. Çıkarım (Inference): Modelin yeni yorumlar üzerinde tahmin yapması ve tahmin sonuçlarının olasılıklarını gösterme

"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import nltk
from nltk.corpus import stopwords

from datasets import load_dataset

# ilk çalıştırmada stop words veri setini indirelim
nltk.download("stopwords")

stop_words_tr = set(stopwords.words("turkish")) # len(stop_words_tr) = 53


# Veri setini yükle

dataset = load_dataset("WhiteAngelss/Turkce-Duygu-Analizi-Dataset")

# 1. Train ve Test parçalarını Pandas'a çevirip alt alta ekleyerek TEK BİR TABLO yap
data = pd.concat([dataset['train'].to_pandas(), dataset['test'].to_pandas()], ignore_index=True)

# 2. Sadece işimize yarayan 2 sütunu al, geri kalan her şeyi çöpe at
data = data[["text", "label"]]

data = data.dropna(subset=["text", "label"])  # Görüş ve Durum sütunlarında eksik veri varsa kaldır

# yorumları ve etiketleri ayır
X = data["text"]  # yorumlar
y = data["label"]  # etiketler (pozitif, negatif, nötr)

def veriyi_temizle(text):

    # string olduğundan emin ol
    if not isinstance(text, str):
        text = str(text)
    # Sayıları kaldır
    text = re.sub(r'\d+', '', text)
    # Küçük harfe çevir
    text = text.lower()
    # Noktalama işaretlerini kaldır
    text = re.sub(r'[^\w\s]', '', text)
    # Stop words'leri kaldır
    text = ' '.join([word for word in text.split() if word not in stop_words_tr])
    return text

X_temiz = X.apply(veriyi_temizle)

X_train, X_test, y_train, y_test = train_test_split(X_temiz, y, test_size=0.2, random_state=42, stratify=y) 
# stratify (katmanlara ayırmak ) parametresi ile eğitim ve test setlerini bölerken y değişkeninen göre 
# adil ve mantıklı bir bölme yapmasını sağlıyoruz
# bölme işlemi tam olarak verideki gerçek pozitif / negatif / nötr oranına göre yapılır

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Unigram ve bigram
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Neden? Özellikle büyük veri setlerinde (mesela 10.000'lerce farklı kelime), 
# matris çok büyür ve modelin eğitimi yavaşlar. Bazen seyrek kelimeler gürültü yaratır.

# Ne işe yarar? TF-IDF skoruna göre en yüksek puana sahip ilk 5000 kelimeyi/öbeği seçer, 
# gerisini atar. Bu hem işlem hızını artırır hem de genellikle modelin aşırı öğrenmesini (overfitting) engeller.

# ngram_range=(1, 2):
# sadece tek kelimelere bakmak bazen yanıltıcıdır.
# "Güzel" (Tek kelime -> Pozitif)
# "Güzel değil" (İki kelime -> Negatif)
# Hem tek kelimeleri (unigram) HEM DE iki kelimelik grupları (bigram) ayrı ayrı özellik olarak al, aynı sepete at

model = LogisticRegression(
    solver='lbfgs',
    C=1.0,
    max_iter=1000,
    random_state=42
) # softmax fonksiyonu kullanır
model.fit(X_train_tfidf, y_train)

accuracy = model.score(X_test_tfidf, y_test)
print(f"Modelin doğruluk oranı: {accuracy:.2f}")

# Bu raddeden sonra Test kısmı yerine yeni yorumlar ekleyip modelin tahmin yapmasını sağlayabiliriz

ornek_yorumlar = [
    "Bu mağaza gerçekten harika, çok memnun kaldım",
    "Berbat bir deneyimdi, asla tavsiye etmem",
    "İdare eder, ne iyi ne kötü"
]

for yorum in ornek_yorumlar:
    yorum_temiz = veriyi_temizle(yorum)
    yorum_tfidf = vectorizer.transform([yorum_temiz])
    tahmin = model.predict(yorum_tfidf)[0]
    olasiliklar = model.predict_proba(yorum_tfidf)[0] # Bu bir matris, tek elemanla çalıştığımız için [0] ile alıyoruz 
    
    print(f"\nYorum: {yorum}")
    print(f"Tahmin edilen duygu: {tahmin}")
    print("Olasılıklar:")
    for sinif, olasilik in zip(model.classes_, olasiliklar): # Etiketler ve olasılıkları karşılıklı eşleştir
        print(f"  {sinif}: {olasilik:.4f}")





