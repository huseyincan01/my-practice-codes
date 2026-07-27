# 🚀 My Practice Codes (Doğal Dil İşleme & Makine Öğrenmesi Pratikleri)

Bu depo, Doğal Dil İşleme (NLP), Derin Öğrenme (Deep Learning) ve Makine Öğrenmesi (Machine Learning) konularında gerçekleştirdiğim uygulama ve projelerin temiz, modüler kodlarını içerir.

---

## 📁 Proje Klasörleri ve İçerikleri

### 1. 📝 `01_text_preprocessing`
* **Dosya:** [`text_preprocessing_guide.py`](./01_text_preprocessing/text_preprocessing_guide.py)
* **Açıklama:** Metin temizleme (regex, URL ve noktalama silme), NLTK ile kelime ve cümle tokenizasyonu, durdurma kelimeleri (stop-words) ayıklama, Stemming (Porter) ve Lemmatization (WordNet) süreçlerini kapsayan modüler rehber.

### 2. 🔤 `02_word_embeddings_clustering`
* **Dosya:** [`word2vec_clustering.py`](./02_word_embeddings_clustering/word2vec_clustering.py)
* **Açıklama:** Gensim Word2Vec mimarisi ile kelime gömmeleri (embeddings) oluşturma, eğitilen vektörleri KMeans kümeleme algoritması ile gruplama ve PCA (Temel Bileşen Analizi) ile 2 boyuta indirgeyip Matplotlib üzerinde görselleştirme.

### 3. 📰 `03_topic_modeling_lda`
* **Dosya:** [`lda_news_modeling.py`](./03_topic_modeling_lda/lda_news_modeling.py)
* **Açıklama:** `scikit-learn` 20 Newsgroups haber veri seti üzerinde CountVectorizer (Bag of Words) ve Latent Dirichlet Allocation (LDA) kullanarak metinlerden gizli konuları (topics) ve en baskın kelimeleri otomatik keşfetme uygulaması.

### 4. 🧠 `04_pytorch_rnn_timeseries`
* **Dosya:** [`pytorch_sine_rnn.py`](./04_pytorch_rnn_timeseries/pytorch_sine_rnn.py)
* **Açıklama:** PyTorch kütüphanesi kullanarak `nn.Module` tabanlı Tekrarlayan Sinir Ağı (RNN) mimarisi tasarımı, zaman serisi veri hazırlığı ve sinüs dalgası üzerinde gelecek adım tahmini.

---

## 🛠️ Kullanılan Teknolojiler
* **Dil:** Python 3.x
* **NLP & Kütüphaneler:** NLTK, Gensim, Scikit-learn, Pandas, NumPy
* **Derin Öğrenme:** PyTorch
* **Görselleştirme:** Matplotlib
