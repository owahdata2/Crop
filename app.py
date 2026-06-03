import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# ================================================
# LOAD DATA
# ================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Crop_recommendation.csv')
    except FileNotFoundError:
        st.error("❌ File 'Crop_recommendation.csv' tidak ditemukan.")
        st.info("Silakan upload file dataset di bawah ini:")
        uploaded_file = st.file_uploader("Upload Crop_recommendation.csv", type=["csv"])
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

# Train Model
model = DecisionTreeClassifier(
    criterion='gini',
    min_samples_leaf=2,
    random_state=42
)
model.fit(X, y_encoded)

# ================================================
# SIDEBAR
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
# MAIN
# ================================================
st.title("🌾 Sistem Rekomendasi Tanaman Pertanian")
st.markdown("**Decision Tree** | Diselaraskan dengan Orange Data Mining")

if st.button("🔍 Prediksi Tanaman", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        'N': N, 'P': P, 'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }])

    pred_idx = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    crop_name = le.inverse_transform([pred_idx])[0]
    confidence = proba[pred_idx] * 100

    st.success(f"**Rekomendasi: {crop_name.upper()}**")
    st.metric("Keyakinan Model", f"{confidence:.1f}%")

    # Top 3
    top3_idx = np.argsort(proba)[::-1][:3]
    st.subheader("Top 3 Rekomendasi")
    for i, idx in enumerate(top3_idx, 1):
        st.write(f"{i}. **{le.inverse_transform([idx])[0]}** — {proba[idx]*100:.1f}%")

# Tabs
tab1, tab2, tab3 = st.tabs(["🌳 Decision Tree", "📈 Feature Importance", "📊 Evaluasi Model"])

with tab1:
    st.subheader("Visualisasi Decision Tree")
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(model, feature_names=X.columns, class_names=le.classes_,
              filled=True, rounded=True, fontsize=8, max_depth=5, ax=ax)
    st.pyplot(fig)

with tab2:
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=X.columns[indices], palette='viridis', ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

with tab3:
    st.subheader("Evaluasi Model (Test Set)")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)  # retrain on train set
    y_pred = model.predict(X_test)
    st.text(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))

# Footer
st.markdown("---")
st.write(f"Dataset: **{len(df)}** sampel | **{len(le.classes_)}** jenis tanaman")
if st.checkbox("Tampilkan 10 data pertama"):
    st.dataframe(df.head(10))

st.caption("Dibuat dengan ❤️ menggunakan Python & Streamlit")
