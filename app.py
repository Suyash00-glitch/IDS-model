import streamlit as st
import pandas as pd
import joblib

# Load saved model, encoder, and scaler
model = joblib.load("marketing_model_rf.pkl")
le = joblib.load("label_encoder.save")
scaler = joblib.load("scaler.save")

st.title("Marketing Response Predictor")
st.write("Enter customer details to predict if they will respond to the campaign.")


# User Inputs
income = st.number_input("Annual Income (₹)", 10000, 10000000, 500000)
credit_score = st.number_input("Credit Score", 300, 850, 700)
product_price = st.number_input("Product Price (₹)", 1000, 1000000, 50000)

# Derived features
affordability = income / product_price
credit_age_ratio = credit_score / 30

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame({
        'Annual_Income_INR': [income],
        'Credit_Score': [credit_score],
        'Product_Price_INR': [product_price],
        'Affordability': [affordability],
        'Credit_Age_Ratio': [credit_age_ratio]
    })

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0]
    label = le.inverse_transform(pred)[0]

    st.subheader(f"Predicted Response: **{label}**")
    st.write(f"✅ Approval Probability: {prob[1]*100:.2f}%")
    st.write(f"❌ Rejection Probability: {prob[0]*100:.2f}%")
