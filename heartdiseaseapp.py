import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="centered"
)

st.title("🫀 Heart Disease Risk Predictor")
st.markdown(
    "Enter patient clinical data below. The model will predict heart disease "
    "risk and explain which factors drove the prediction."
)
st.divider()

with st.sidebar:
    st.header("Model Info")
    st.metric("Model", "Random Forest")
    st.metric("Test ROC-AUC", "0.9044")
    st.metric("Test Accuracy", "82.1%")
    st.metric("Dataset", "UCI Heart Disease")
    st.metric("Patients", "920")
    st.markdown("---")
    st.caption(
        "Model trained on 4 clinical centers: Cleveland, Hungary, "
        "Switzerland, VA Long Beach. Features selected based on "
        "clinical relevance and SHAP importance analysis."
    )

st.subheader("Patient Information")
col1, col2 = st.columns(2)

with col1:
    age      = st.number_input("Age", min_value=20, max_value=100, value=55)
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=220, value=130)
    chol     = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=650, value=230)
    thalch   = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=140)
    oldpeak  = st.number_input("ST Depression (oldpeak)", min_value=-3.0, max_value=7.0, value=0.5, step=0.1)

with col2:
    sex     = st.selectbox("Sex", ["Male", "Female"])
    cp      = st.selectbox("Chest Pain Type", ["asymptomatic", "atypical angina", "non-anginal", "typical angina"])
    fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [False, True])
    restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"])
    exang   = st.selectbox("Exercise Induced Angina", [False, True])
    slope   = st.selectbox("ST Slope", ["flat", "upsloping", "downsloping"])
    ca      = st.selectbox("Number of Major Vessels (0-3)", [0.0, 1.0, 2.0, 3.0])
    thal    = st.selectbox("Thalassemia", ["normal", "reversable defect", "fixed defect", "Missing"])

st.divider()

def build_input(age, sex, cp, trestbps, chol, fbs, restecg,
                thalch, exang, oldpeak, slope, ca, thal):
    raw = pd.DataFrame([{
        'age': age, 'trestbps': trestbps, 'chol': chol,
        'thalch': thalch, 'oldpeak': oldpeak,
        'sex': sex, 'cp': cp, 'fbs': fbs,
        'restecg': restecg, 'exang': exang,
        'slope': slope, 'ca': ca, 'thal': thal
    }])
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    raw_encoded = pd.get_dummies(raw, columns=categorical_features, drop_first=False)
    expected_cols = [
        'age', 'trestbps', 'chol', 'thalch', 'oldpeak',
        'sex_Female', 'sex_Male',
        'cp_asymptomatic', 'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
        'fbs_False', 'fbs_True',
        'restecg_lv hypertrophy', 'restecg_normal', 'restecg_st-t abnormality',
        'exang_False', 'exang_True',
        'slope_downsloping', 'slope_flat', 'slope_upsloping',
        'ca_0.0', 'ca_1.0', 'ca_2.0', 'ca_3.0',
        'thal_Missing', 'thal_fixed defect', 'thal_normal', 'thal_reversable defect'
    ]
    for col in expected_cols:
        if col not in raw_encoded.columns:
            raw_encoded[col] = 0
    return raw_encoded[expected_cols]


def clean_name(feature):
    replacements = {
        'cp_asymptomatic':        'Asymptomatic chest pain',
        'cp_atypical angina':     'Atypical angina chest pain',
        'cp_non-anginal':         'Non-anginal chest pain',
        'cp_typical angina':      'Typical angina chest pain',
        'exang_True':             'Exercise-induced angina present',
        'exang_False':            'No exercise-induced angina',
        'thal_normal':            'Normal thalassemia',
        'thal_reversable defect': 'Reversable thalassemia defect',
        'thal_fixed defect':      'Fixed thalassemia defect',
        'slope_flat':             'Flat ST slope',
        'slope_upsloping':        'Upsloping ST slope',
        'slope_downsloping':      'Downsloping ST slope',
        'fbs_True':               'High fasting blood sugar',
        'fbs_False':              'Normal fasting blood sugar',
        'restecg_normal':         'Normal resting ECG',
        'ca_0.0':                 'No major vessels affected',
        'ca_1.0':                 '1 major vessel affected',
        'ca_2.0':                 '2 major vessels affected',
        'ca_3.0':                 '3 major vessels affected',
        'sex_Male':               'Male sex',
        'sex_Female':             'Female sex',
        'age':                    'Age',
        'thalch':                 'Maximum heart rate',
        'oldpeak':                'ST depression',
        'chol':                   'Cholesterol level',
        'trestbps':               'Resting blood pressure',
    }
    return replacements.get(feature, feature.replace('_', ' ').title())


if st.button("Predict Risk", type="primary", use_container_width=True):

    X_input = build_input(age, sex, cp, trestbps, chol, fbs,
                          restecg, thalch, exang, oldpeak, slope, ca, thal)

    model  = joblib.load('best_rf_model.pkl')
    scaler = joblib.load('robust_scaler.pkl')

    continuous_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    X_input[continuous_features] = scaler.transform(X_input[continuous_features])

    prob = model.predict_proba(X_input)[0][1]
    pred = int(prob >= 0.5)

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"⚠️ High Risk — {prob:.1%} probability of heart disease")
    else:
        st.success(f"✅ Low Risk — {prob:.1%} probability of heart disease")

    st.progress(float(prob))
    st.caption(f"Risk score: {prob:.3f} | Threshold: 0.500")

    st.subheader("Why this prediction?")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_input)
    sv        = shap_vals[:, :, 1]
    base_val  = explainer.expected_value[1]

    fig, ax = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(
            values=sv[0],
            base_values=base_val,
            data=X_input.iloc[0].values,
            feature_names=X_input.columns.tolist()
        ), show=False
    )
    st.pyplot(fig)
    plt.close()

    st.subheader("Explanation")

    feature_names    = X_input.columns.tolist()
    shap_vals_single = sv[0]
    pairs            = list(zip(feature_names, shap_vals_single))
    pairs_sorted     = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
    pushing_up       = [(f, v) for f, v in pairs_sorted if v > 0.01][:3]
    pushing_down     = [(f, v) for f, v in pairs_sorted if v < -0.01][:3]

    if prob >= 0.7:
        risk_label = "high"
    elif prob >= 0.5:
        risk_label = "moderate"
    else:
        risk_label = "low"

    top_feature, top_val = pairs_sorted[0]
    direction = "increasing" if top_val > 0 else "reducing"

    up_names   = [clean_name(f) for f, v in pushing_up]
    down_names = [clean_name(f) for f, v in pushing_down]
    up_text    = ", ".join(up_names)   if up_names   else "none identified"
    down_text  = ", ".join(down_names) if down_names else "none identified"

    st.markdown(f"""
This patient has a **{risk_label} risk** of heart disease with a predicted probability of **{prob:.1%}**.

The prediction is primarily driven by **{clean_name(top_feature)}**, which is the strongest factor
{direction} the risk for this patient. Other notable factors raising the risk include {up_text},
while factors working in the patient's favour include {down_text}.

The model bases this on patterns learned from 920 patients across 4 clinical centers.
A score above 50% indicates likely presence of heart disease.
    """)

    st.warning(
        "⚕️ This prediction is for informational purposes only and is not a "
        "medical diagnosis. Always consult a qualified healthcare professional."
    )

st.divider()
st.caption("Built with Streamlit · Model: Random Forest (UCI Heart Disease) · Explainability: TreeSHAP")
