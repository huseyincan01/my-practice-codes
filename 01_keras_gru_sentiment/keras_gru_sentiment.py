"""
Keras & GRU ile IMDB Film Yorumları Duygu Analizi (Sentiment Analysis)
-----------------------------------------------------------------------
Bu çalışma, Keras kütüphanesinin hazır IMDB veri kümesini kullanarak
GRU (Gated Recurrent Unit) katmanı tabanlı bir Derin Öğrenme sınıflandırma modeli oluşturur.
Model, metin dizilerini pad_sequences ile eşit uzunluğa getirir ve duygu tahmini (Pozitif/Negatif) yapar.
"""

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

def build_and_train_gru():
    num_words = 10000        # En sık kullanılan 10.000 kelime
    max_seq_length = 200     # Her yorum maksimum 200 token
    embedding_dim = 100

    print("IMDB Veri seti yükleniyor...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

    # Padding (Tüm yorumları 200 uzunluğuna sabitleme)
    X_train_padded = pad_sequences(X_train, maxlen=max_seq_length)
    X_test_padded = pad_sequences(X_test, maxlen=max_seq_length)

    # Keras GRU Mimarisi
    model = Sequential([
        Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=max_seq_length),
        GRU(units=64, return_sequences=False),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\n--- Model Özeti ---")
    model.summary()

    # Model Eğitimi
    print("\nKeras GRU Modeli Eğitiliyor...")
    model.fit(
        X_train_padded,
        y_train,
        epochs=3,
        batch_size=256,
        validation_split=0.2,
        verbose=1
    )

    # Test Değerlendirmesi
    loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    print(f"\nTest Başarımı - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    build_and_train_gru()
