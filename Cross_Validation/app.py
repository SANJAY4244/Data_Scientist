import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Cross Validation App", page_icon="🔁", layout="centered")

st.title("🔁 Cross Validation App")
st.write("Logistic Regression with K-Fold Cross Validation")

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Cross_Validation.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset loaded successfully")
    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # --------------------------------------------------
    # Feature & Target split
    # --------------------------------------------------
    if "target" not in df.columns:
        st.error("❌ Dataset must contain a 'target' column")
    else:
        X = df.drop("target", axis=1)
        y = df["target"]

        st.write("### 🔍 Feature Columns")
        st.write(list(X.columns))

        # --------------------------------------------------
        # K-Fold input
        # --------------------------------------------------
        k = st.slider("Select number of folds (K)", min_value=2, max_value=10, value=5)

        # --------------------------------------------------
        # Model & Cross Validation
        # --------------------------------------------------
        model = LogisticRegression(max_iter=1000)

        kf = KFold(
            n_splits=k,
            shuffle=True,
            random_state=42
        )

        scores = cross_val_score(
            model,
            X,
            y,
            cv=kf,
            scoring="accuracy"
        )

        # --------------------------------------------------
        # Results
        # --------------------------------------------------
        st.write("### ✅ Cross Validation Results")
        st.write("Fold Accuracies:", scores)
        st.write("Mean Accuracy:", scores.mean())
        st.write("Standard Deviation:", scores.std())

        st.success(f"🎯 Final Accuracy: {scores.mean() * 100:.2f}%")
