import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and scaler
model = joblib.load('linear_regression_model.joblib')
scaler = joblib.load('scaler.pkl')

st.set_page_config(layout="wide")
st.title('Insurance Charge Prediction')

st.markdown("### Enter Patient Details for Charges Prediction")

# Create input fields for the features
# Use st.columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    claim_amount = st.number_input('Claim Amount (USD)', min_value=0.0, value=1000.0, step=100.0)
    past_consultations = st.number_input('Past Consultations', min_value=0, value=2, step=1)

with col2:
    hospital_expenditure = st.number_input('Hospital Expenditure (USD)', min_value=0.0, value=500.0, step=50.0)
    annual_salary = st.number_input('Annual Salary (USD)', min_value=0.0, value=50000.0, step=1000.0)

with col3:
    children = st.number_input('Number of Children', min_value=0, value=1, step=1)
    smoker_option = st.selectbox('Smoker?', ('No', 'Yes'))
    smoker = 1 if smoker_option == 'Yes' else 0

# Create a button to predict
if st.button('Predict Charges'):
    # Define the exact numerical columns the scaler was fitted on, including 'charges'
    numerical_cols_for_scaler = ['claim_amount', 'past_consultations', 'hospital_expenditure', 'annual_salary', 'children', 'smoker', 'charges']

    # Prepare the input data for the scaler, including a dummy 'charges' column
    input_data_for_scaler = pd.DataFrame([[claim_amount, past_consultations, hospital_expenditure, annual_salary, children, smoker, 0.0]],
                                         columns=numerical_cols_for_scaler)

    # Scale the full input data
    scaled_full_data = scaler.transform(input_data_for_scaler)

    # Get the index of 'charges' within the numerical_cols_for_scaler
    charges_idx = numerical_cols_for_scaler.index('charges')

    # Extract only the scaled feature columns (excluding 'charges') for model prediction
    # The model was trained on X (features without charges), so it expects n_features - 1 columns.
    scaled_input_features_for_model = np.delete(scaled_full_data, charges_idx, axis=1)

    # Make prediction using the model (which predicts the scaled 'charges')
    prediction_scaled = model.predict(scaled_input_features_for_model)

    # Create a dummy array for inverse transformation
    dummy_input_for_inverse = np.zeros((1, len(numerical_cols_for_scaler)))
    dummy_input_for_inverse[0, charges_idx] = prediction_scaled[0] # Place the scaled prediction at the correct index

    # Inverse transform the dummy array to get the prediction in original scale (USD)
    original_scale_values = scaler.inverse_transform(dummy_input_for_inverse)
    predicted_charges_usd = original_scale_values[0, charges_idx]

    # Convert to INR (assuming 1 USD = 83 INR)
    predicted_charges_inr = predicted_charges_usd * 83

    st.success(f'Predicted Insurance Charges (USD): **${predicted_charges_usd:,.2f}**')
    st.success(f'Predicted Insurance Charges (INR): **₹{predicted_charges_inr:,.2f}**')
