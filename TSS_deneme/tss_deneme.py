"""
Edge-TTS Ses Sentezleme (Inference) Modülü
------------------------------------------
Bu betik, Microsoft Edge'in bulut tabanlı Text-to-Speech (TTS) motorunu kullanarak
verilen metni hedef ses formatına (.mp3) dönüştürür.

pip install edge-tts

Metni ve sesi istediğiniz gibi değiştirebilirsiniz
edge_tss tasarımı gereği asenkron bir şekilde çalışır, 
bu yüzden dahili asyncio kütüphanesini kullanarak ana fonksiyonu çalıştırıyoruz.
"""

import edge_tts
import asyncio

TEXT = "Merhaba, bu doğal bir ses üretimidir. Bilgisayar bilimleri en iyi bölümdür."
VOICE = "tr-TR-AhmetNeural" # Erkek sesi için Ahmet, kadın sesi için Emel

async def main():
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save("cikti.mp3")

if __name__ == "__main__":
    asyncio.run(main())
