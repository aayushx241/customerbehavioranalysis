
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Title
st.title("Customer Segmentation Tool")

# File Upload Section
st.sidebar.header("Upload your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(data)
    
    # Feature Selection
    st.sidebar.header("Clustering Options")
    features = st.sidebar.multiselect("Select features for clustering", data.columns)
    
    if len(features) > 1:
        X = data[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Select Number of Clusters
        n_clusters = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        data['Cluster'] = clusters
        st.write("### Clustered Data")
        st.dataframe(data)
        
        # Visualization
        st.write("### Cluster Visualization")
        if len(features) >= 2:
            fig = px.scatter(data, x=features[0], y=features[1], color=data['Cluster'].astype(str),
                             title="Customer Segmentation", labels={'color': 'Cluster'})
            st.plotly_chart(fig)
        else:
            st.warning("Select at least two features for visualization.")
        
        # Download Segmented Data
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Clustered Data", data=csv, file_name="segmented_data.csv", mime="text/csv")
    else:
        st.warning("Please select at least two features for clustering.")
else:
    st.info("Awaiting CSV file to be uploaded. Upload a file to get started!")

st.sidebar.markdown("*")
