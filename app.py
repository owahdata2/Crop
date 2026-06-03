import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# ================================================
# LOAD DATA & MODEL/ENCODER SIAP PAKAI
# ================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Crop_recommendation.csv')
        return df
    except FileNotFoundError:
        st.error("❌ File 'Crop_recommendation.csv' tidak ditemukan di repositori GitHub Anda.")
        st.stop()

@st.cache_resource
def load_artifacts():
    try:
        # Memuat model dan kamus label encoder yang sudah diekspor dari skrip training Anda
        model = joblib.load('crop_model_orange.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, le
    except FileNotFoundError as e:
        st.error(f"❌ Artefak model tidak lengkap: {e.filename} tidak ditemukan.")
        st.info("Pastika file 'crop_model_orange.pkl' dan 'label_encoder.pkl' sudah diupload ke GitHub.")
        st.stop()

df = load_data()
model, le = load_artifacts()

X = df.drop('label', axis=1)
# Menggunakan encoder pkl asli untuk memastikan konsistensi target class
y_encoded = le.transform(df['label']) 

# ================================================
# SIDEBAR - INPUT PARAMETER
# ================================================
st.sidebar.header("🌱 Input Parameter")
N = st.sidebar.slider("Nitrogen (N)", 0, 140, 60)
P = st.sidebar.slider("Phosphorus (P)", 5, 145, 50)
K = st.sidebar.slider("Potassium (K)", 5, 205, 40)
temperature = st.sidebar.slider("Temperature (°C)", 8.0, 45.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 70.0)
ph = st.sidebar.slider("pH Tanah", 3.5, 10.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

# ================================================
# MAIN UTAMA
# ================================================
st.title("🌾 Sistem Rekomendasi Tanaman Pertanian")
st.markdown("**Decision Tree** | Terintegrasi dengan Model Produksi `crop_model_orange.pkl`")

# Pembagian kolom hasil prediksi
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Hasil Analisis")
    if st.button("🔍 Prediksi Tanaman", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            'N': N, 'P': P, 'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }])

        # Prediksi menggunakan model produksi langsung
        pred_idx = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        crop_name = le.inverse_transform([pred_idx])[0]
        confidence = proba[pred_idx] * 100

        st.success(f"### **Rekomendasi Utama: {crop_name.upper()}**")
        st.metric("Tingkat Keyakinan Model", f"{confidence:.2f}%")

        # Top 3 Rekomendasi Alternatif
        top3_idx = np.argsort(proba)[::-1][:3]
        st.write("---")
        st.markdown("💡 **Top 3 Rekomendasi Alternatif:**")
        for i, idx in enumerate(top3_idx, 1):
            st.write(f"{i}. **{le.inverse_transform([idx])[0].title()}** — ({proba[idx]*100:.1f}%)")

with col2:
    # Menggunakan tab untuk memisahkan visualisasi berat agar performa web cepat
    tab1, tab2, tab3 = st.tabs(["📊 Matriks Evaluasi", "📈 Feature Importance", "🌳 Struktur Aturan Pohon"])

    with tab1:
        st.markdown("### Ringkasan Validasi Model Utama")
        # Menggunakan prediksi keseluruhan data aktual untuk menghasilkan visualisasi yang konsisten
        y_all_pred = model.predict(X)
        
        # Plot Confusion Matrix yang Dioptimalkan untuk Streamlit
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        cm = confusion_matrix(y_encoded, y_all_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=False, cmap='Blues', 
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
        ax_cm.set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
        plt.setp(ax_cm.get_xticklabels(), rotation=90, ha='right', fontsize=8)
        plt.setp(ax_cm.get_yticklabels(), fontsize=8)
        st.pyplot(fig_cm)
        
        # Menampilkan Classification Report Berbasis Teks
        st.markdown("**Classification Report Lengkap:**")
        st.text(classification_report(y_encoded, y_all_pred, target_names=le.classes_, digits=4))

    with tab2:
        st.markdown("### Tingkat Pengaruh Parameter (Feature Importance)")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
        sns.barplot(x=importances[indices], y=X.columns[indices], palette='viridis', ax=ax_fi)
        for i, v in enumerate(importances[indices]):
            ax_fi.text(v + 0.01, i, f"{v:.3f}", va='center', fontweight='bold', fontsize=9)
        ax_fi.set_xlim(0, max(importances) + 0.05)
        st.pyplot(fig_fi)

    with tab3:
        st.markdown("### Struktur Kedalaman Model Pohon")
        st.info(f"Pohon Keputusan memiliki total **{model.tree_.node_count}** simpul (nodes) dengan kedalaman maksimum **{model.tree_.max_depth}** tingkatan.")
        
        # Karena plot_tree terlalu bertumpuk untuk max_depth=None, kita tampilkan pratinjau grafik berukuran sedang
        with st.expander("👁️ Tampilkan Grafik Pratinjau Pohon (Terbatas Level 3)"):
            fig_tree, ax_tree = plt.subplots(figsize=(15, 8))
            from sklearn.tree import plot_tree
            plot_tree(model, feature_names=X.columns, class_names=le.classes_,
                      filled=True, rounded=True, fontsize=7, max_depth=3, ax=ax_tree)
            st.pyplot(fig_tree)
            st.caption("Catatan: Grafik di atas dibatasi sampai kedalaman 3 tingkat untuk kenyamanan visualisasi di web.")

# ================================================
# FOOTER & INSPEKSI DATA
# ================================================
st.markdown("---")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.write(f"Total Baris Dataset: **{len(df)}** sampel | Terdaftar: **{len(le.classes_)}** Komoditas Tanaman")
with col_f2:
    if st.checkbox("Lihat Sampel Data"):
        st.dataframe(df.head(10))

st.caption("Dibuat dengan ❤️ menggunakan Python & Streamlit Community Cloud")
