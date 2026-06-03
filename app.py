import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# ================================================
# LOAD DATA (Lebih Aman untuk Streamlit Cloud)
# ================================================
@st.cache_data
def load_data():
    try:
        # Path untuk lokal / workdir
        df = pd.read_csv('/home/workdir/attachments/Crop_recommendation.csv')
    except FileNotFoundError:
        try:
            # Path alternatif untuk Streamlit Cloud
            df = pd.read_csv('Crop_recommendation.csv')
        except FileNotFoundError:
            st.error("❌ File dataset tidak ditemukan. Silakan upload file CSV.")
            uploaded_file = st.file_uploader("Upload Crop_recommendation.csv", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
            else:
                st.stop()
    return df

df = load_data()

X = df.drop('label', axis=1)
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train Model (kita train ulang agar tidak bergantung ke file .pkl)
model = DecisionTreeClassifier(
    criterion='gini',
    min_samples_leaf=2,
    random_state=42
)
model.fit(X, y_encoded)

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
st.markdown("**Decision Tree** | *Diselaraskan dengan Orange Data Mining*")

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

        # Top 3
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
tab1, tab2, tab3 = st.tabs(["📈 Feature Importance", "🌳 Decision Tree", "📊 Info Model"])

with tab1:
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=X.columns[indices], ax=ax, palette='viridis')
    ax.set_title("Feature Importance")
    st.pyplot(fig)

with tab2:
    st.subheader("Struktur Decision Tree")
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(model, feature_names=X.columns, 
              class_names=le.classes_, filled=True, 
              rounded=True, fontsize=7, max_depth=5, ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader("Evaluasi Model")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))

# ================================================
# FOOTER
# ================================================
st.markdown("---")
st.write(f"Total data: **{len(df)}** sampel | **{len(le.classes_)}** jenis tanaman")
if st.checkbox("Tampilkan Contoh Data"):
    st.dataframe(df.head(10))

st.caption("Dibuat dengan ❤️ | Decision Tree • Diselaraskan dengan Orange Data Mining")
