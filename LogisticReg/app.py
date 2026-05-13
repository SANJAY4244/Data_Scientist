import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Logistic Regression App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Logistic Regression Classification App")
st.write("Upload a CSV file and perform Logistic Regression")

# -------------------------------
# Upload CSV
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.warning("⚠️ Please upload a CSV file to continue")
    st.stop()

# Load dataset
df = pd.read_csv(uploaded_file)
st.success("✅ Dataset loaded successfully")
st.dataframe(df.head(), use_container_width=True)

# -------------------------------
# Select Target Column
# -------------------------------
target_col = st.selectbox("🎯 Select Target Column", df.columns)

if target_col is None:
    st.stop()

X = df.drop(columns=[target_col])
y = df[target_col]

# -------------------------------
# Encode categorical data
# -------------------------------
le = LabelEncoder()

for col in X.select_dtypes(include=["object"]):
    X[col] = le.fit_transform(X[col])

if y.dtype == "object":
    y = le.fit_transform(y)

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Feature scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Train Logistic Regression
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# Accuracy
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.success(f"Accuracy: {accuracy:.2f}")

# -------------------------------
# Prediction Section
# -------------------------------
st.subheader("🔮 Make a Prediction")

input_data = []

for col in X.columns:
    value = st.number_input(
        f"{col}",
        value=float(df[col].mean())
    )
    input_data.append(value)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"✅ Predicted Class: **{prediction[0]}**")
