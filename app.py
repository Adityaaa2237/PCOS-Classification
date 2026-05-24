import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title = "Deteksi PCOS",
    page_icon  = "🔬",
    layout     = "centered"
)

# ============================================================
# LOAD MODEL (hanya sekali saat pertama jalan)
# ============================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_vit_pcos_oversampling.h5')
    return model

model = load_model()

# ============================================================
# FUNGSI PREPROCESSING
# Sama persis seperti preprocessing di notebook kamu
# ============================================================
def preprocess_image(image_bytes):
    # Baca gambar dari bytes
    img_array = np.frombuffer(image_bytes, np.uint8)
    img       = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    # Resize ke 224x224
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Denoising (median filter 5x5) seperti di notebook
    img = cv2.medianBlur(img, 5)

    # Normalisasi 0-1
    img = img / 255.0

    # Tambah dimensi batch dan channel → (1, 224, 224, 1)
    img = np.expand_dims(img, axis=(0, -1)).astype(np.float32)

    return img

# ============================================================
# TAMPILAN WEB APP
# ============================================================

# Judul
st.title("🔬 Deteksi PCOS")
st.markdown("**Sistem Deteksi Polycystic Ovary Syndrome (PCOS) menggunakan Vision Transformer**")
st.divider()

# Upload gambar
st.subheader("Upload Citra USG")
uploaded_file = st.file_uploader(
    "Pilih file gambar USG (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Tampilkan gambar yang diupload
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Citra USG yang Diupload**")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    # Preprocessing & Prediksi
    image_bytes = uploaded_file.read()
    img_processed = preprocess_image(image_bytes)

    # Tombol prediksi
    st.divider()
    if st.button("🔍 Analisis Gambar", use_container_width=True, type="primary"):

        with st.spinner("Sedang menganalisis gambar..."):
            # Prediksi
            prob  = model.predict(img_processed, verbose=0)[0][0]
            label = 1 if prob >= 0.5 else 0

        # Tampilkan hasil
        st.divider()
        st.subheader("Hasil Analisis")

        if label == 1:
            st.error("## ⚠️ PCOS Terdeteksi")
            st.markdown(f"""
            Berdasarkan analisis citra USG menggunakan model Vision Transformer,
            **gambar ini terindikasi PCOS (Polycystic Ovary Syndrome).**
            """)
        else:
            st.success("## ✅ Tidak Terdeteksi PCOS")
            st.markdown(f"""
            Berdasarkan analisis citra USG menggunakan model Vision Transformer,
            **gambar ini tidak terindikasi PCOS (Polycystic Ovary Syndrome).**
            """)

        # Probabilitas
        st.divider()
        col3, col4 = st.columns(2)

        with col3:
            st.metric(
                label = "Probabilitas PCOS",
                value = f"{prob*100:.2f}%"
            )
        with col4:
            st.metric(
                label = "Probabilitas Non-PCOS",
                value = f"{(1-prob)*100:.2f}%"
            )

        # Progress bar
        st.markdown("**Tingkat Keyakinan Model:**")
        st.progress(float(prob) if label == 1 else float(1 - prob))

        # Disclaimer
        st.divider()
        st.warning("""
        ⚠️ **Disclaimer:** Hasil analisis ini hanya bersifat sebagai alat bantu 
        dan tidak menggantikan diagnosis medis dari dokter spesialis. 
        Konsultasikan hasil ini dengan tenaga medis profesional.
        """)

else:
    # Tampilan awal sebelum upload
    st.info("👆 Silakan upload citra USG untuk memulai analisis.")

    st.divider()
    st.markdown("### Cara Penggunaan")
    st.markdown("""
    1. Klik tombol **Browse files** di atas
    2. Pilih file gambar USG format JPG atau PNG
    3. Klik tombol **Analisis Gambar**
    4. Tunggu hasil analisis ditampilkan
    """)

    st.markdown("### Tentang Sistem")
    st.markdown("""
    Sistem ini menggunakan arsitektur **Vision Transformer (ViT)** 
    yang dilatih menggunakan dataset citra USG PCOS untuk mendeteksi 
    ada atau tidaknya indikasi PCOS pada citra USG ovarium.
    """)
