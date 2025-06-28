import streamlit as st
import pandas as pd
import joblib

# Load saved objects
model = joblib.load('SVM_heart_decease.pkl')
scalar = joblib.load('Scalar.pkl')
colms = joblib.load('Columns.pkl')

st.title('Heart Decease Prediction App ❤ ❤')
st.markdown('Provide the following details to check your heart decease risk:')

# Collect user input
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):

    # Build raw input dictionary based on encoding used during training
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        f'Sex_{sex}': 1,
        f'ChestPainType_{chest_pain}': 1,
        f'RestingECG_{resting_ecg}': 1,
        f'ExerciseAngina_{exercise_angina}': 1,
        f'ST_Slope_{st_slope}': 1
    }

    # Create dataframe
    input_df = pd.DataFrame([raw_input])

    # Ensure all expected columns exist; fill missing with 0
    for col in colms:
        if col not in input_df.columns:
            input_df[col] = 0

    # Drop any unexpected columns
    input_df = input_df[colms]

    # Scale input
    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
    input_df[num_cols] = scalar.transform(input_df[num_cols])

    # Predict
    prediction = model.predict(input_df)[0]

    # Display result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Decease")
    else:
        st.success("✅ Low Risk of Heart Decease")
