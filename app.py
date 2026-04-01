import streamlit as st
import pandas as pd
import joblib

# Load the model you uploaded
model = joblib.load('house_price_model.joblib')

st.set_page_config(page_title="House Price Predictor", page_icon="🏡")

st.title("🏡 House Price Predictor")
st.write("Enter the details below to estimate the house price.")

# User inputs
location = st.selectbox("Location", ['Downtown', 'Suburb', 'Rural'])
size = st.number_input("Size (in sqft)", min_value=500, max_value=10000, value=1500)
rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=3)

# Prediction logic
if st.button("Predict Price"):
    # Creating a small dataframe for the model to read
    input_data = pd.DataFrame({
        'Location': [location],
        'Size_sqft': [size],
        'Rooms': [rooms]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display the result formatted as currency
    st.success(f"The estimated price is: ${prediction:,.2f}")
  
