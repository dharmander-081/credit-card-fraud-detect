import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

st.title("💳 Credit Card Fraud Detection")
st.markdown("""
This application uses a Machine Learning model to detect whether a credit card transaction is **Legitimate** or **Fraudulent**.
Please enter the transaction features below to test the model.
""")

st.markdown("### Transaction Details")
st.markdown("Typing 30 features is tedious! Use these buttons to auto-fill the form with real examples:")

col_a, col_b = st.columns(2)

# Hardcoded examples
sample_legit = {
    'Time': 0.0, 'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155, 
    'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698, 'V9': 0.363787, 
    'V10': 0.090794, 'V11': -0.551600, 'V12': -0.617801, 'V13': -0.991390, 'V14': -0.311169, 
    'V15': 1.468177, 'V16': -0.470401, 'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 
    'V20': 0.251412, 'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928, 
    'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053, 'Amount': 149.62
}

sample_fraud = {
    'Time': 406.0, 'V1': -2.312227, 'V2': 1.951992, 'V3': -1.609851, 'V4': 3.997906, 
    'V5': -0.522188, 'V6': -1.426545, 'V7': -2.537387, 'V8': 1.391657, 'V9': -2.770089, 
    'V10': -2.772272, 'V11': 3.202033, 'V12': -2.899907, 'V13': -0.595222, 'V14': -4.289254, 
    'V15': 0.389724, 'V16': -1.140747, 'V17': -2.830056, 'V18': -0.016822, 'V19': 0.416956, 
    'V20': 0.126911, 'V21': 0.517232, 'V22': -0.035049, 'V23': -0.465211, 'V24': 0.320198, 
    'V25': 0.044519, 'V26': 0.177840, 'V27': 0.261145, 'V28': -0.143276, 'Amount': 0.00
}

if col_a.button("🟢 Auto-fill Legitimate Transaction"):
    st.session_state['data'] = sample_legit

if col_b.button("🔴 Auto-fill Fraudulent Transaction"):
    st.session_state['data'] = sample_fraud

# Initialize feature inputs
features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
input_data = {}
current_data = st.session_state.get('data', {f: 0.0 for f in features})

# Display inputs in an columns to make it compact
col1, col2, col3 = st.columns(3)

columns = [col1, col2, col3]

for i, feature in enumerate(features):
    col = columns[i % 3]
    with col:
        input_data[feature] = st.number_input(
            f"{feature}", 
            value=float(current_data[feature]),
            format="%.6f"
        )

# Prediction button
st.markdown("---")
if st.button("Predict Transaction Status", type="primary", use_container_width=True):
    # Prepare input for model
    input_df = pd.DataFrame([input_data])
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    st.markdown("### Prediction Result")
    if prediction == 0:
        st.success("✅ The model predicts this is a **Legitimate** transaction.")
    else:
        st.error("🚨 The model predicts this is a **Fraudulent** transaction.")
