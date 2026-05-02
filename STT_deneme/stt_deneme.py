"""
   ================================================================================
   PROJE: Çoklu Ses Formatı STT (Speech-to-Text) Analiz Testi
   MODEL: OpenAI Whisper (Small)
   CİHAZ: NVIDIA CUDA GPU Hızlandırma
   ================================================================================

   AÇIKLAMA:
   Bu kod, aynı ses içeriğinin farklı dijital formatlarda (WAV, OPUS, MP3) 
   saklanmasının, Whisper modelinin transkripsiyon doğruluğu üzerindeki etkisini 
   ölçmek için tasarlanmıştır. Fakat (bulamadığım için) doğrudan WAV formatında kaydedilmiş bir dosya yerine, 
   YouTube'dan alınan bir ses klibinin WAV formatına dönüştürürerek kullandım. Normal şartlarda WAV formatı 
   ham ve kayıpsız olduğu için en yüksek olması gerekir, burdaki sahte WAV olduğu için öyle olmaz.


   TEKNİK DETAYLAR:
   1. Model Seçimi: 'small' modeli 241M parametreye sahip olup, hız ve doğruluk 
      dengesi (WER - Word Error Rate) açısından prototipleme için optimize edilmiştir.
   2. CUDA Kullanımı: Model, CPU yerine GPU (CUDA) üzerinde çalıştırılarak 
      VRAM üzerinde paralel hesaplama (tensors) yapılması sağlanmıştır.
   3. Dil Kilidi: 'language="tr"' parametresi, modelin dil tahminleme (language 
      identification) aşamasını atlayarak doğrudan Türkçe fonetik ve semantik 
      çözümlemeye odaklanmasını sağlar.

   FORMAT ANALİZİ (Öngörülen):

   -  WAV: Ham veya kayıpsız veridir
   -  OPUS: YouTube'un yerel ses formatıdıt; insan sesi karakteristiklerini düşük bit 
   hızlarında bile en iyi koruyan (bitrate-to-quality) modern format olduğu için 
   genellikle STT modellerinde en tutarlı sonucu verir.

   -  MP3: Psikoakustik (insanın sesi nasıl algıladığını inceleyen bilim) modelleme 
   ile bazı frekansları (insan kulağının duymadığı) atar; bu durum 
   AI modellerinin öznitelik çıkarımını bazen negatif etkileyebilir.

pip install openai-whisper
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

   ffmpeg kurulması gerekmektedir
   Normalde MP3 veya OP3 gibi formatlar, sesi "paketleyip" sıkıştırır. 
   FFMPEG ise bu paketlemeyi çözer ve sesi bilgisayarın en saf haliyle 
   işleyebileceği bir sayı dizisine yani ham sese (bu sürecce PCM - Pulse Code Modulation denir) dönüştürür.

pip install static-ffmpeg

   pip ile kurmayı sağlayan bir kütüphane imiş bu, 
   FFmpeg’in bilgisayara manuel olarak kurulması, "PATH" ayarlarının yapılması ve 
   işletim sistemi uyumluluğu gibi dertleri ortadan kaldıran bir Python kütüphanesi.

   kullanımı:
   import static_ffmpeg

   # Bu satır kritik yerdir: 
   # FFmpeg'i bulur, yoksa indirir ve geçici olarak sistem yoluna (PATH) ekler.
   static_ffmpeg.add_paths()

"""

import whisper

model = whisper.load_model("small", device="cuda")
# small model bizim için yeterlidir
# cuda kullanarak modeli GPU üzerinde çalıştırıyoruz

print("İşte başlıyoruz")

print("\n\nWAV format: ")
result = model.transcribe("kesit.wav", language="tr")
print(result["text"])

print("\n\nOPUS format: ")
result1 = model.transcribe("kesit.opus", language="tr")
print(result1["text"])

print("\n\nMP3 format: ")
result2 = model.transcribe("kesit.mp3", language="tr")
print(result2["text"])