import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


st.set_page_config(page_title="Hierarchical Clustering", layout="wide")
st.title("🌳 Hierarchical Clustering App")


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least 2 numerical columns.")
    else:
        st.subheader("🔢 Select Features for Clustering")
        selected_features = st.multiselect(
            "Choose columns",
            numeric_cols,
            default=numeric_cols[1:3]
        )

        if len(selected_features) >= 2:
            X = df[selected_features]

            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Dendrogram
            st.subheader("🌿 Dendrogram")
            linked = linkage(X_scaled, method="ward")

            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(linked, ax=ax)
            plt.xlabel("Data Points")
            plt.ylabel("Distance")
            st.pyplot(fig)

            # Number of clusters
            n_clusters = st.slider("Select number of clusters", 2, 10, 3)

            # Hierarchical Clustering
            model = AgglomerativeClustering(n_clusters=n_clusters)
            df["Cluster"] = model.fit_predict(X_scaled)

            st.subheader("📊 Clustered Data")
            st.dataframe(df)

            st.success("Hierarchical Clustering completed successfully ✅")
        else:
            st.warning("Please select at least 2 features for clustering.")
else:
    st.info("👆 Upload a CSV file to get started.")
