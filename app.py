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

# Initialize feature inputs
features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
input_data = {}

# Display inputs in an columns to make it compact
col1, col2, col3 = st.columns(3)

columns = [col1, col2, col3]

for i, feature in enumerate(features):
    col = columns[i % 3]
    with col:
        input_data[feature] = st.number_input(
            f"{feature}", 
            value=0.0,
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
