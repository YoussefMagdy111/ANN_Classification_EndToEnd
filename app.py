import streamlit as st
import import_ipynb  
from prediction import predict  # Make sure prediction.py has a predict() function

# Streamlit Page Setup
st.set_page_config(page_title="ANN Classifier", layout="centered")
st.title("🔮 ANN Customer Prediction")
st.write("Fill in the details below to predict whether a customer will churn.")

# --- Input Form ---
with st.form("prediction_form"):
    credit_score = st.number_input("Credit Score", min_value=300, max_value=1000, value=600)
    age = st.slider("Age", min_value=18, max_value=100, value=35)
    tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=3)
    balance = st.number_input("Account Balance", min_value=0.0, step=100.0, value=60000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    salary = st.number_input("Estimated Salary", min_value=0.0, step=100.0, value=50000.0)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.radio("Gender", ["Male", "Female"])
    has_card = st.radio("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x else "No")
    is_active = st.radio("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x else "No")

    submit = st.form_submit_button("Predict")

# --- Run Prediction ---
if submit:
    # Assemble feature dict in correct order
    features = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "EstimatedSalary": salary,
        "Geography": geography,
        "Gender": gender,
        "HasCrCard": has_card,
        "IsActiveMember": is_active
    }

    label, probability = predict(features)

    st.markdown("### 🧾 Prediction Result")
    st.success(f"Prediction: **{label}**")
    st.info(f"Probability: **{probability:.2%}**")
