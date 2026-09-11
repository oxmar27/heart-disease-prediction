
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_disease_model.pkl")

st.title("Heart Disease Prediction")

st.write("Enter the patient's information below.")

age = st.number_input("Age", min_value=1, max_value=100, value=50)

sex = st.selectbox("Sex", [0, 1])

cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=80,
    max_value=220,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=200
)

fbs = st.selectbox("Fasting Blood Sugar", [0, 1])

restecg = st.selectbox("Resting ECG", [0, 1, 2])

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=220,
    value=150
)

exang = st.selectbox("Exercise Induced Angina", [0, 1])

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=7.0,
    value=1.0
)

slope = st.selectbox("Slope", [0, 1, 2])

ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])

thal = st.selectbox("Thal", [3, 6, 7])

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("The model predicts a higher likelihood of heart disease.")
    else:
        st.success("The model predicts a lower likelihood of heart disease.")

    st.write(f"Prediction probability: {probability:.2%}")
