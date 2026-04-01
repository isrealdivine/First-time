import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Lagos House Predictor", page_icon="🇳🇬")

# --- LOAD DATA AND TRAIN MODEL ---
@st.cache_resource
def load_and_train():
    # 1. Read your uploaded CSV
    df = pd.read_csv('Lagos_houses_prices.csv')
    
    # 2. Select the features we want to use
    # We will use Location, Bedrooms, Bathrooms, and Prop_Type
    features = ['Location', 'Bedrooms', 'Bathrooms', 'Prop_Type', 'Is_Island']
    X = df[features]
    y = df['Price']
    
    # 3. Create a preprocessor for text (Location and Prop_Type)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Location', 'Prop_Type'])
        ],
        remainder='passthrough'
    )
    
    # 4. Build the Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # 5. Train the model
    model.fit(X, y)
    
    # Return everything needed for the UI
    return model, df['Location'].unique(), df['Prop_Type'].unique()

# Run the training function
model, locations, prop_types = load_and_train()

# --- STREAMLIT USER INTERFACE ---
st.title("🏡 Lagos House Price Predictor")
st.write("Predict the price of houses in Lagos using real market data!")

# Create the inputs
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Select Location", sorted(locations))
    prop_type = st.selectbox("Property Type", sorted(prop_types))
    is_island = st.radio("Is it on the Island?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=20, value=3)
    bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=20, value=3)

# Prediction button
if st.button("Calculate Estimated Price"):
    # Create input dataframe
    input_df = pd.DataFrame({
        'Location': [location],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Prop_Type': [prop_type],
        'Is_Island': [is_island]
    })
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    # Show the result in Naira
    st.success(f"The estimated price is: ₦{max(0, prediction):,.2f}")
    st.info("Note: This is an estimate based on recent Lagos housing data.")
