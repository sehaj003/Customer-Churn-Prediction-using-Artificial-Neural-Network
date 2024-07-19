import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Load the model and scaler
model = load_model('model.keras')
sc = joblib.load('scaler.pkl')

# Streamlit app title
st.title('Customer Churn Prediction')

# Function to make predictions
def predict_churn(input_data):
    input_df = pd.DataFrame([input_data])
    columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
               'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Germany', 'Spain', 'Male']
    input_scaled = sc.transform(input_df[columns])
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]
    return prediction_proba

# Streamlit form for user input
with st.form(key='churn_form'):
    CreditScore = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
    Age = st.number_input('Age', min_value=18, max_value=100, value=40)
    Tenure = st.number_input('Tenure (years)', min_value=0, max_value=10, value=3)
    Balance = st.number_input('Balance', min_value=0.0, value=60000.0)
    NumOfProducts = st.number_input('Number of Products', min_value=1, max_value=4, value=2)
    HasCrCard = st.selectbox('Has Credit Card', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    IsActiveMember = st.selectbox('Is Active Member', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    EstimatedSalary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
    Germany = st.selectbox('Is from Germany', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    Spain = st.selectbox('Is from Spain', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    Male = st.selectbox('Is Male', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

    submit_button = st.form_submit_button(label='Predict Churn')

# Handle form submission
if submit_button:
    input_data = {
        'CreditScore': CreditScore,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary,
        'Germany': Germany,
        'Spain': Spain,
        'Male': Male
    }
    prediction_proba = predict_churn(input_data)
    st.write(f"Prediction Probability: {prediction_proba:.2f}")
    if prediction_proba > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')
