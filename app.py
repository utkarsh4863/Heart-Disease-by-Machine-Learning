import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Load Model
# ----------------------------

model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    color: #e63946;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #457b9d;
    font-size: 18px;
}

.stButton > button {
    background-color: #e63946;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 50px;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Title
# ----------------------------

st.markdown(
    "<p class='title'>❤️ Heart Disease Prediction App</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Predict Heart Disease Risk using Machine Learning</p>",
    unsafe_allow_html=True
)

# ----------------------------
# Inputs
# ----------------------------

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        20,
        80,
        50
    )

    resting_bp = st.slider(
        "Resting Blood Pressure",
        80,
        220,
        120
    )

    cholesterol = st.slider(
        "Cholesterol",
        50,
        650,
        200
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        140
    )

    oldpeak = st.slider(
        "Oldpeak",
        0.0,
        6.5,
        1.0
    )

with col2:

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["No", "Yes"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY"]
    )

    ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    

    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["No", "Yes"]
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

# ----------------------------
# Encoding
# ----------------------------

sex_m = 1 if sex == "Male" else 0

cp_ata = 1 if chest_pain == "ATA" else 0
cp_nap = 1 if chest_pain == "NAP" else 0
cp_ta = 1 if chest_pain == "TA" else 0

ecg_normal = 1 if ecg == "Normal" else 0
ecg_st = 1 if ecg == "ST" else 0

exercise_angina_y = 1 if exercise_angina == "Yes" else 0

st_slope_flat = 1 if st_slope == "Flat" else 0
st_slope_up = 1 if st_slope == "Up" else 0

fasting_bs = 1 if fasting_bs == "Yes" else 0

# ----------------------------
# Predict Button
# ----------------------------

if st.button("Predict"):

    input_df = pd.DataFrame([[
    age,
    resting_bp,
    cholesterol,
    fasting_bs,
    max_hr,
    oldpeak,
    sex_m,
    cp_ata,
    cp_nap,
    cp_ta,
    ecg_normal,
    ecg_st,
    exercise_angina_y,
    st_slope_flat,
    st_slope_up
]], columns=[
    'Age',
    'RestingBP',
    'Cholesterol',
    'FastingBS',
    'MaxHR',
    'Oldpeak',
    'Sex_M',
    'ChestPainType_ATA',
    'ChestPainType_NAP',
    'ChestPainType_TA',
    'RestingECG_Normal',
    'RestingECG_ST',
    'ExerciseAngina_Y',
    'ST_Slope_Flat',
    'ST_Slope_Up'
])
    num_cols = [
        'Age',
        'RestingBP',
        'Cholesterol',
        'MaxHR',
        'Oldpeak'
    ]

    input_df[num_cols] = scaler.transform(
        input_df[num_cols]
    )

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")