import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Predict house price using **Random Forest Regression**")

# -----------------------------------
# Load Dataset
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("house_price_regression.csv")

df = load_data()

st.success("Dataset loaded successfully!")
st.dataframe(df.head(), use_container_width=True)

# -----------------------------------
# Features & Target
# -----------------------------------
X = df.drop("House_Price", axis=1)
y = df["House_Price"]

# -----------------------------------
# Train-Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# Model Training
# -----------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------------
# Model Accuracy
# -----------------------------------
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"R² Score: **{score:.2f}**")

# -----------------------------------
# User Inputs
# -----------------------------------
st.subheader("🏡 Enter House Details")

area = st.number_input("Area (sqft)", min_value=300, max_value=5000, value=1200)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)
distance = st.number_input("Distance to City (km)", min_value=1, max_value=50, value=7)


if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms, parking, distance]])
    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated House Price: ₹ {prediction:,.0f}")
