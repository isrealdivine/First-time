import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Lagos Real Estate AI", page_icon="🇳🇬", layout="wide")

@st.cache_resource
def load_and_train():
    df = pd.read_csv('Lagos_houses_prices.csv')
    # Using more features for better accuracy
    features = ['Location', 'Bedrooms', 'Bathrooms', 'Prop_Type', 'Has_Pool', 'Has_BQ']
    X = df[features]
    y = df['Price']
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Location', 'Prop_Type'])],
        remainder='passthrough'
    )
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    model.fit(X, y)
    return model, df

model, full_df = load_and_train()

st.title("🇳🇬 Lagos House Price AI Predictor")
st.markdown("---")

# Layout with two columns
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Configure Property")
    loc = st.selectbox("Location", sorted(full_df['Location'].unique()))
    p_type = st.selectbox("Property Type", sorted(full_df['Prop_Type'].unique()))
    
    beds = st.slider("Bedrooms", 1, 10, 3)
    baths = st.slider("Bathrooms", 1, 10, 3)
    
    # New Luxury Features
    pool = st.checkbox("Has Swimming Pool?")
    bq = st.checkbox("Has Boys Quarters (BQ)?")

if st.button("Predict Market Value", use_container_width=True):
    input_df = pd.DataFrame({
        'Location': [loc], 'Bedrooms': [beds], 'Bathrooms': [baths], 
        'Prop_Type': [p_type], 'Has_Pool': [1 if pool else 0], 'Has_BQ': [1 if bq else 0]
    })
    
    res = model.predict(input_df)[0]
    
    with right_col:
        st.metric("Estimated Price", f"₦{res:,.2f}")
        # Fun formatting
        if res >= 1000000:
            st.write(f"💡 That's approximately **₦{res/1000000:.1f} Million**")

with right_col:
    st.subheader("Market Heatmap")
    # Show a map of houses in the selected location
    map_data = full_df[full_df['Location'] == loc][['lat', 'lon']]
    st.map(map_data)
