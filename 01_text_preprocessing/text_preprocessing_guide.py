"""
Metin Ön İşleme Rehberi (Text Preprocessing Guide)
-------------------------------------------------
Bu modül, Doğal Dil İşleme (NLP) projelerinde yaygın olarak kullanılan
metin temizleme, tokenizasyon, kök/gövde bulma (stemming/lemmatization)
ve etkisiz kelimeleri (stop words) ayıklama adımlarını derli toplu sunar.
"""

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Gerekli NLTK paketlerinin indirilmesi
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_text(text: str) -> str:
    """Metindeki özel karakterleri, noktalama işaretlerini ve fazla boşlukları temizler."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # URL temizleme
    text = re.sub(r'[^\w\s]', '', text)                                    # Noktalama temizleme
    text = re.sub(r'\d+', '', text)                                         # Sayıları temizleme
    text = text.strip()
    return text

def tokenize_text(text: str):
    """Metni kelimelerine ve cümlelerine ayırır."""
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    return words, sentences

def remove_stopwords(words: list, lang: str = 'english') -> list:
    """Belirtilen dildeki stop-word (etkisiz) kelimeleri listeden çıkarır."""
    stop_words = set(stopwords.words(lang))
    filtered_words = [w for w in words if w.lower() not in stop_words]
    return filtered_words

def apply_stemming_and_lemmatization(words: list):
    """Kelimelere Stemming (Porter) ve Lemmatization (WordNet) uygular."""
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    stemmed = [stemmer.stem(w) for w in words]
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    
    return stemmed, lemmatized

if __name__ == "__main__":
    sample_text = "Hello World! NLP is amazing. We are learning text preprocessing with NLTK and Python at 2026."
    
    print("--- Orijinal Metin ---")
    print(sample_text)
    
    cleaned = clean_text(sample_text)
    print("\n--- Temizlenmiş Metin ---")
    print(cleaned)
    
    words, sents = tokenize_text(cleaned)
    print(f"\n--- Tokenizasyon ({len(words)} Kelime, {len(sents)} Cümle) ---")
    print("Kelimeler:", words)
    
    filtered = remove_stopwords(words)
    print("\n--- Stop Words Temizlendi ---")
    print(filtered)
    
    stemmed, lemmatized = apply_stemming_and_lemmatization(filtered)
    print("\n--- Stemming ---")
    print(stemmed)
    print("\n--- Lemmatization ---")
    print(lemmatized)
