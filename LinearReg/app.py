import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Salary Prediction App")

# Load dataset
df = pd.read_csv("Salary_dataset.csv")

# Correct features and target
X = df[["YearsExperience"]]
y = df["Salary"]

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
years = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.5
)

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict([[years]])
    st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")
