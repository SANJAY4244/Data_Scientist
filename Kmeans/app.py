import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ------------------------------------
# Page Config
# ------------------------------------
st.set_page_config(
    page_title="K-Means Clustering App",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Segmentation using K-Means")
st.write("Unsupervised Learning – K-Means Clustering")

# ------------------------------------
# Load Dataset
# ------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Kmeans.csv")

df = load_data()

st.success("Dataset loaded successfully")
st.dataframe(df.head(), use_container_width=True)

# ------------------------------------
# Feature Selection
# ------------------------------------
X = df.drop("Customer_ID", axis=1)

# ------------------------------------
# Feature Scaling
# ------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------
# Choose Number of Clusters
# ------------------------------------
k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=4)

# ------------------------------------
# Train K-Means Model
# ------------------------------------
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ------------------------------------
# Cluster Count
# ------------------------------------
st.subheader("🔢 Cluster Distribution")
st.write(df["Cluster"].value_counts())

# ------------------------------------
# Visualization
# ------------------------------------
st.subheader("📈 Cluster Visualization")

fig, ax = plt.subplots()
scatter = ax.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"]
)

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segmentation")

st.pyplot(fig)

# ------------------------------------
# Cluster Centers
# ------------------------------------
st.subheader("🎯 Cluster Centers (Scaled)")
st.write(kmeans.cluster_centers_)
