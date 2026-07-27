"""
20 Newsgroups Haber Veri Seti ile LDA Konu Modelleme (Topic Modeling)
----------------------------------------------------------------------
Bu modül, scikit-learn ve NLTK kullanarak haber metinlerini temizler,
CountVectorizer ile vektörleştirir ve Latent Dirichlet Allocation (LDA)
algoritması ile gizli konu başlıklarını (topics) otomatik olarak keşfeder.
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

def clean_news_text(text: str, stop_words: set) -> str:
    """Haber metinlerini küçük harfe çevirir, noktalama ve stop-word'leri kaldırır."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def run_lda_topic_modeling(n_topics: int = 2):
    stop_words = set(stopwords.words('english'))
    categories = ['rec.sport.baseball', 'rec.sport.hockey', 'sci.electronics', 'sci.med']
    
    print("20 Newsgroups veri seti indiriliyor...")
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )
    
    print(f"Yüklenen Toplam Haber: {len(newsgroups.data)}")
    
    # Metin Temizleme
    cleaned_docs = [clean_news_text(doc, stop_words) for doc in newsgroups.data]
    
    # Vektörleştirme (Bag of Words)
    vectorizer = CountVectorizer(max_df=0.9, min_df=5)
    X = vectorizer.fit_transform(cleaned_docs)
    
    # LDA Eğitimi
    print(f"LDA modeli {n_topics} konu için eğitiliyor...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=30,
        random_state=42,
        learning_method="batch"
    )
    lda.fit(X)
    
    feature_names = vectorizer.get_feature_names_out()
    print("\n--- Keşfedilen Konular ve En Önemli Kelimeler ---")
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        print(f"Konu {topic_idx + 1}: {', '.join(top_words)}")
        
    doc_topic_dist = lda.transform(X)
    df_results = pd.DataFrame(doc_topic_dist, columns=[f"Konu_{i+1}_Olasilik" for i in range(n_topics)])
    df_results["Gercek_Kategori"] = [newsgroups.target_names[i] for i in newsgroups.target]
    
    print("\n--- Örnek Doküman Konu Dağılımı ---")
    print(df_results.head(5).round(3))

if __name__ == "__main__":
    run_lda_topic_modeling(n_topics=2)
