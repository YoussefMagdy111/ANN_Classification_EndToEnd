import streamlit as st
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model 

# Load the trained model
model = load_model('model.h5')


# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app UI
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title(' Customer Churn Prediction')

# User inputs
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92, 35)
balance = st.number_input('Balance', value=50000.0, step=100.0)
credit_score = st.number_input('Credit Score', value=650, min_value=300, max_value=1000)
estimated_salary = st.number_input('Estimated Salary', value=60000.0, step=100.0)
tenure = st.slider('Tenure', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Encode categorical values manually
gender_encoded = 1 if gender == 'Male' else 0  # Assuming Male = 1, Female = 0

# One-hot encoding for Geography
# Assume: France = [1, 0, 0], Germany = [0, 1, 0], Spain = [0, 0, 1]
geo_dict = {
    'France': [1, 0, 0],
    'Germany': [0, 1, 0],
    'Spain': [0, 0, 1]
}
geo_encoded = geo_dict[geography]

# Combine all inputs in correct order as used during training
input_features = [
    credit_score,
    gender_encoded,
    age,
    tenure,
    balance,
    num_of_products,
    has_cr_card,
    is_active_member,
    estimated_salary
] + geo_encoded  # Add geography one-hot

# Convert to DataFrame and scale
input_df = pd.DataFrame([input_features])
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)[0][0]

# Show results
st.markdown("### Prediction Result")
st.write(f' **Churn Probability:** `{prediction:.2%}`')

if prediction >= 0.5:
    st.error(" The customer is **likely to churn**.")
else:
    st.success(" The customer is **not likely to churn**.")
