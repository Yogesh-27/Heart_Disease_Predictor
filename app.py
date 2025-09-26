import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")
# scaler = joblib.load("scaler.pkl")  # uncomment if used

# Page config
st.set_page_config(page_title="Heart Disease Predictor ❤️", layout="wide")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below and check the prediction.")

# Layout with columns
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 1, 120, 30)
    sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)

with col2:
    chol = st.slider("Serum Cholestoral (chol)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2])
    thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 250, 150)

with col3:
    exang = st.radio("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of the peak exercise ST segment", options=[0, 1, 2])
    ca = st.selectbox("Number of major vessels (ca)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

# Prepare features
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

# If scaler used:
# features = scaler.transform(features)

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(features)[0]

    st.subheader("📊 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ Heart Disease Likely")
        st.markdown("The model predicts that the patient is **at risk of heart disease.** 🚑")
    else:
        st.success("✅ No Heart Disease")
        st.markdown("The model predicts that the patient is **not likely to have heart disease.** 🫀")
