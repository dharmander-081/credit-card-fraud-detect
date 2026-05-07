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

# Load the model and dataset
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('creditcard_sample.csv')

model = load_model()
df = load_data()

st.title("💳 Credit Card Fraud Detection")
st.markdown("""
This application uses a Machine Learning model to detect whether a credit card transaction is **Legitimate** or **Fraudulent**.
Since entering 30 features manually is tedious, you can use the sidebar to load a random sample from the dataset.
""")

st.sidebar.header("Testing Options")
st.sidebar.markdown("Use a random sample from the dataset to test the model.")

# Options to select random sample
sample_type = st.sidebar.radio("Select Sample Type:", ("Legitimate (0)", "Fraudulent (1)", "Random"))

if st.sidebar.button("Load Sample Data"):
    if sample_type == "Legitimate (0)":
        sample = df[df.Class == 0].sample(1)
    elif sample_type == "Fraudulent (1)":
        sample = df[df.Class == 1].sample(1)
    else:
        sample = df.sample(1)
        
    st.session_state['sample_data'] = sample.iloc[0].to_dict()

st.markdown("### Transaction Details")

# Initialize feature inputs
features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
input_data = {}

# Use session state or default to 0.0
sample_data = st.session_state.get('sample_data', {feature: 0.0 for feature in features})
actual_class = sample_data.get('Class', None)

if actual_class is not None:
    st.info(f"Loaded a sample transaction. Actual Class in dataset: **{'Fraudulent' if actual_class == 1 else 'Legitimate'}**")

# Display inputs in an expander or columns (we'll use columns to make it compact)
col1, col2, col3 = st.columns(3)

columns = [col1, col2, col3]

for i, feature in enumerate(features):
    col = columns[i % 3]
    with col:
        input_data[feature] = st.number_input(
            f"{feature}", 
            value=float(sample_data.get(feature, 0.0)),
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
