import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# 1. Configuration and Model Loading
st.set_page_config(page_title="Churn Sentinel AI", layout="wide")

# Correct paths to reach the Backend folder from the Frontend folder
MODEL_PATH = '../Backend/Model/churn_model.pkl'
FEATURES_PATH = '../Backend/Model/features_list.pkl'

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ model.pkl not found! Please run model_trainer.py first.")
        return None, None
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

model, feature_cols = load_artifacts()

# 2. UI Header
st.title("🛡️ Churn Sentinel: Real-Time Retention Tool")
st.markdown("Enter customer details below to predict churn risk and see retention strategies.")

# 3. Sidebar for Manual Input
st.sidebar.header("Customer Profile")
def get_user_input():
    # These match the features expected by your trained model
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    contract = st.sidebar.selectbox("Contract Type", [0, 1, 2], 
                                    help="0: Month-to-month, 1: One year, 2: Two year")
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0, 150, 70)
    total_charges = st.sidebar.number_input("Total Charges ($)", 0, 8000, 1000)
    
    # Matching the 'Service_Count' feature engineered in your processor
    service_count = st.sidebar.slider("Number of Add-on Services", 0, 6, 2)
    
    # Building the input dictionary
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'Service_Count': service_count,
        'Charge_Density': total_charges / (tenure + 1)
    }
    
    # Ensure all other model features (like gender, internet service) are present
    # We default them to 0 for this manual entry tool
    df = pd.DataFrame(data, index=[0])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    return df[feature_cols] # Return columns in the exact order model expects

# 4. Execution Logic
if model:
    input_df = get_user_input()
    
    if st.button("Analyze Risk Profile"):
        # Get probability of class 1 (Churn)
        prediction_proba = model.predict_proba(input_df)[0][1]
        
        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Churn Probability", f"{prediction_proba:.1%}")
            
        with col2:
            if prediction_proba > 0.7:
                st.error("🚨 HIGH RISK CUSTOMER")
                st.write("**Strategy:** Offer a high-value loyalty discount or a 2-year contract upgrade.")
            elif prediction_proba > 0.4:
                st.warning("⚠️ MODERATE RISK CUSTOMER")
                st.write("**Strategy:** Suggest bundling 'Tech Support' or 'Online Security' to increase stickiness.")
            else:
                st.success("✅ LOW RISK CUSTOMER")
                st.write("**Strategy:** Standard engagement; no immediate intervention needed.")

        # Optional: Show the data being sent to the model
        with st.expander("See Raw Input Vector"):
            st.dataframe(input_df)