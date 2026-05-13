import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


st.set_page_config(page_title="Hierarchical Clustering", layout="wide")
st.title("🍷 Wine Dataset – Hierarchical Clustering")


uploaded_file = st.file_uploader(
    "📁 Upload CSV file (Wine dataset)",
    type=["csv"]
)

if uploaded_file is not None:
 
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

   
    X = df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

   
    
    st.sidebar.header("⚙️ Clustering Settings")
    n_clusters = st.sidebar.slider(
        "Select number of clusters",
        min_value=2,
        max_value=6,
        value=3
    )

   
    st.subheader("🌳 Dendrogram")

    linked = linkage(X_scaled, method="ward")

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linked, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Distance")

    st.pyplot(fig)

    
    hc = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",
        linkage="ward"
    )

    clusters = hc.fit_predict(X_scaled)
    df["Cluster"] = clusters

    st.subheader("📌 Clustered Data")
    st.dataframe(df, use_container_width=True)

    st.subheader("📈 Cluster Distribution")
    st.bar_chart(df["Cluster"].value_counts())

    st.subheader("🔍 Cluster Visualization")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    scatter = ax2.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=clusters
    )

    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.set_title("Hierarchical Clustering Result")

    st.pyplot(fig2)

else:
    st.info("👆 Please upload a CSV file to begin clustering")
