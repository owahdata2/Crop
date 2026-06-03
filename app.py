import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

# ================================================
# CONFIGURATION
# ================================================
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# Load Model & Encoder
@st.cache_resource
def load_model():
    model = joblib.load('/home/workdir/attachments/crop_fruit_model_orange_style.pkl')
    encoder = joblib.load('/home/workdir/attachments/label_encoder.pkl')
    return model, encoder

model, le = load_model()

# Load Dataset untuk referensi
df = pd.read_csv('/home/workdir/attachments/Crop_recommendation.csv')

# ================================================
# SIDEBAR
# ================================================
st.sidebar.header("🌱 Crop Recommendation AI")
st.sidebar.image("https://img.icons8.com/fluency/96/000000/plant.png", width=100)

st.sidebar.markdown("### Masukkan Parameter Tanah & Cuaca")

N = st.sidebar.slider("Nitrogen (N)", 0, 140, 60)
P = st.sidebar.slider("Phosphorus (P)", 5, 145, 50)
K = st.sidebar.slider("Potassium (K)", 5, 205, 40)
temperature = st.sidebar.slider("Temperature (°C)", 8.0, 45.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 70.0)
ph = st.sidebar.slider("pH Tanah", 3.5, 10.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

# ================================================
# MAIN PAGE
# ================================================
st.title("🌾 Sistem Rekomendasi Tanaman Pertanian")
st.markdown("**Decision Tree** | *Menggunakan parameter tanah & cuaca untuk merekomendasikan tanaman terbaik*")

col1, col2 = st.columns([3, 2])

with col1:
    if st.button("🔍 Prediksi Tanaman", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            'N': N, 'P': P, 'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        crop_name = le.inverse_transform([prediction])[0]
        confidence = probability[prediction] * 100

        st.success(f"**Rekomendasi: {crop_name.upper()}**")
        st.metric("Tingkat Keyakinan", f"{confidence:.1f}%")

        # Top 3 rekomendasi
        top3_idx = np.argsort(probability)[::-1][:3]
        st.subheader("Top 3 Rekomendasi")
        for i, idx in enumerate(top3_idx, 1):
            prob = probability[idx] * 100
            crop = le.inverse_transform([idx])[0]
            st.write(f"{i}. **{crop}** — {prob:.1f}%")

with col2:
    st.subheader("📊 Parameter Saat Ini")
    param_df = pd.DataFrame({
        'Parameter': ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
        'Nilai': [N, P, K, temperature, humidity, ph, rainfall]
    })
    st.table(param_df)

# ================================================
# VISUALISASI
# ================================================
tab1, tab2, tab3 = st.tabs(["📈 Feature Importance", "🌳 Decision Tree", "📊 Confusion Matrix"])

with tab1:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=df.columns[:-1][indices], ax=ax, palette='viridis')
    ax.set_title("Feature Importance")
    st.pyplot(fig)

with tab2:
    st.subheader("Struktur Decision Tree (max_depth = 5)")
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(model, feature_names=df.columns[:-1], 
              class_names=le.classes_, filled=True, 
              rounded=True, fontsize=8, max_depth=5, ax=ax)
    st.pyplot(fig)

with tab3:
    st.info("Confusion Matrix dari evaluasi model (Test Set)")
    st.image("https://i.imgur.com/placeholder.png", caption="Confusion Matrix akan ditampilkan jika ada file gambar")

# ================================================
# INFORMASI DATASET
# ================================================
st.markdown("---")
st.subheader("📋 Informasi Dataset")
st.write(f"Total data: **{len(df)}** sampel | **22** jenis tanaman")

if st.checkbox("Tampilkan Contoh Data"):
    st.dataframe(df.head(10))

# Footer
st.caption("Dibuat dengan ❤️ menggunakan Decision Tree | Diselaraskan dengan Orange Data Mining")