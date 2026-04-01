import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="House Price Predictor", page_icon="🏡")

# --- INTERNAL MODEL TRAINING ---
# This part replaces the .joblib file so you don't get errors!
@st.cache_resource
def train_model():
    data = {
        'Location': ['Downtown', 'Suburb', 'Rural', 'Downtown', 'Suburb', 'Rural', 'Downtown', 'Suburb'],
        'Size_sqft': [1500, 2000, 1200, 1800, 2500, 1000, 2200, 1900],
        'Rooms': [3, 4, 2, 3, 5, 2, 4, 3],
        'Price': [450000, 350000, 150000, 500000, 420000, 120000, 600000, 330000]
    }
    df = pd.DataFrame(data)
    X = df[['Location', 'Size_sqft', 'Rooms']]
    y = df['Price']
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Location'])],
        remainder='passthrough'
    )
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    model_pipeline.fit(X, y)
    return model_pipeline

model = train_model()
# --- END TRAINING ---

st.title("🏡 House Price Predictor")
st.write("Estimate house prices based on simple features.")

location = st.selectbox("Location", ['Downtown', 'Suburb', 'Rural'])
size = st.number_input("Size (in sqft)", min_value=500, max_value=10000, value=1500)
rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=3)

if st.button("Predict Price"):
    input_data = pd.DataFrame({'Location': [location], 'Size_sqft': [size], 'Rooms': [rooms]})
    prediction = model.predict(input_data)[0]
    st.success(f"The estimated price is: ${max(0, prediction):,.2f}")
