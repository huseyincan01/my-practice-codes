"""
Word2Vec Kelime Gömülmesi (Word Embedding) ve KMeans Kümeleme
-------------------------------------------------------------
Bu çalışma, metin verisi üzerinden Gensim Word2Vec kullanarak 
kelime vektörleri temsil etmeyi, eğitilen vektörleri KMeans ile kümelemeyi
ve PCA (Principal Component Analysis) kullanarak 2 boyutta görselleştirmeyi gösterir.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

def metin_temizleme(metin: str, stop_words: set) -> str:
    """Metni küçük harfe çevirir, noktalama ve sayıları kaldırır."""
    metin = metin.lower()
    metin = re.sub(r"\d+", "", metin)
    metin = re.sub(r"[^\w\s]", "", metin)
    
    kelimeler = metin.split()
    kelimeler = [k for k in kelimeler if k not in stop_words and len(k) > 2]
    return " ".join(kelimeler)

def train_and_visualize_word2vec(sample_texts: list):
    stop_words = set(stopwords.words("english"))
    
    # Metin temizleme & tokenizasyon
    temiz_metinler = [metin_temizleme(t, stop_words) for t in sample_texts]
    tokenize_cumleler = [simple_preprocess(t) for t in temiz_metinler]
    
    # Word2Vec Model Eğitimi
    print("Word2Vec modeli eğitiliyor...")
    model = Word2Vec(
        sentences=tokenize_cumleler,
        vector_size=50,
        window=5,
        min_count=1,
        sg=0
    )
    
    kelime_vektorleri = model.wv
    kelimeler = list(kelime_vektorleri.index_to_key)[:100] # Görselleştirme için ilk 100 kelime
    vektorler = [kelime_vektorleri[w] for w in kelimeler]
    
    # KMeans ile 2 Kümeleme
    kmeans = KMeans(n_clusters=2, random_state=42)
    kume_etiketleri = kmeans.fit_predict(vektorler)
    
    # PCA ile 2 Boyuta İndirgeme
    pca = PCA(n_components=2)
    indirgenmis_vektorler = pca.fit_transform(vektorler)
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    plt.scatter(indirgenmis_vektorler[:, 0], indirgenmis_vektorler[:, 1], c=kume_etiketleri, cmap="viridis", alpha=0.7)
    
    # Küme merkezleri
    merkezler = pca.transform(kmeans.cluster_centers_)
    plt.scatter(merkezler[:, 0], merkezler[:, 1], c="red", marker="X", s=200, label="Küme Merkezleri")
    
    for i, kelime in enumerate(kelimeler[:30]): # İlk 30 kelime etiketini ekranda göster
        plt.annotate(kelime, (indirgenmis_vektorler[i, 0], indirgenmis_vektorler[i, 1]), fontsize=9)
        
    plt.title("Word2Vec + KMeans Kümeleme + PCA (2D Görselleştirme)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    output_path = "word2vec_clusters.png"
    plt.savefig(output_path)
    print(f"Görselleştirme kaydedildi: {output_path}")

if __name__ == "__main__":
    # Örnek cümle grubu (Veri seti bulunamadığında çalışabilmesi için)
    demo_corpus = [
        "The movie was absolutely fantastic and brilliant! Great acting and director.",
        "Worst film I have ever seen. Terrible script, bad acting, complete garbage.",
        "An amazing cinematic experience, breathtaking visuals and great music.",
        "Horrible waste of time. I hated the plot and the characters were annoying.",
        "Deep learning and artificial intelligence are revolutionizing natural language processing."
    ]
    train_and_visualize_word2vec(demo_corpus)
