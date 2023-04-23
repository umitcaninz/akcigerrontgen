import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Modeli yükleme
model = load_model('model.h5')

# Sınıfların etiketleri
classes = ['NORMAL', 'PNEUMONIA']

# Görüntü işleme fonksiyonu
def process_image(img):
    # Görüntü boyutunu yeniden boyutlandırma
    resized = cv2.resize(img, (150, 150))
    # Normalleştirme
    normalized = resized / 255.0
    # Boyut değiştirme ve şekil eklemek
    reshaped = np.reshape(normalized, (1, 150, 150, 3))
    # Sonuçları döndürme
    return reshaped

# Streamlit uygulaması
def main():
    st.title("Akciğer Kanseri Tespit Uygulaması")
    # Dosya yükleyicisi ekleme
    uploaded_file = st.file_uploader("Lütfen bir akciğer röntgeni yükleyin", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Dosyayı okuma
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        # Görüntüyü işleme
        processed_image = process_image(image)
        # Tahmin yapma
        prediction = model.predict(processed_image)[0][0]
        predicted_class = classes[int(round(prediction))]
        # Sonucu gösterme
        st.write("Bu görüntü ", predicted_class, " sınıfına aittir.")
        st.image(image, caption="Yüklenen Görüntü", use_column_width=True)

if __name__ == '__main__':
    main()

