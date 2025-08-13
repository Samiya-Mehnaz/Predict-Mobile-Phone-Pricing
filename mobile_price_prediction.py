import streamlit as st
import joblib
import numpy as np

# Load model and feature names
model = joblib.load("mobile_price_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("📱 Mobile Phone Price Range Prediction")
st.write("Enter the phone specifications below to predict the price range.")

# Create input fields dynamically
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, value=0.0)
    user_input.append(value)

# Convert to numpy array
user_input = np.array(user_input).reshape(1, -1)

if st.button("Predict"):
    prediction = model.predict(user_input)[0]
    price_labels = {
        0: "Low Cost 💰",
        1: "Medium Cost 💵",
        2: "High Cost 💎",
        3: "Very High Cost 🏆"
    }
    st.success(f"Predicted Price Range: **{price_labels[prediction]}**")
