
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Procurement Anomaly Detection", layout="wide")
st.title("🚨 Procurement Fraud Anomaly Detection Dashboard")

# Upload Excel File
uploaded_file = st.file_uploader("Upload Procurement Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    original_df = df.copy()

    # Drop unneeded columns
    df_proc = df.drop(columns=['Transaction_ID', 'Invoice_Description'], errors='ignore')

    # Fill missing values
    df_proc = df_proc.fillna(0)

    # Encode all categorical columns
    cat_cols = df_proc.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df_proc[col] = LabelEncoder().fit_transform(df_proc[col].astype(str))

    # Scale only numeric columns
    numeric_cols = df_proc.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_proc[numeric_cols])

    # Apply Isolation Forest
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(df_scaled)
    df['Anomaly_Score'] = model.decision_function(df_scaled)
    df['Anomaly'] = model.predict(df_scaled)
    df['Anomaly_Label'] = df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})

    # Sidebar Filters
    st.sidebar.header("🔎 Filter Options")
    selected_department = st.sidebar.multiselect("Department", df['Department'].unique(), default=df['Department'].unique())
    selected_method = st.sidebar.multiselect("Procurement Method", df['Procurement_Method'].unique(), default=df['Procurement_Method'].unique())

    filtered_df = df[df['Department'].isin(selected_department) & df['Procurement_Method'].isin(selected_method)]

    # Alert Threshold
    alert_threshold = st.sidebar.slider("Alert Threshold (Anomaly Score)", float(df['Anomaly_Score'].min()), float(df['Anomaly_Score'].max()), float(df['Anomaly_Score'].quantile(0.05)))

    st.subheader("📋 Filtered Procurement Transactions")
    st.dataframe(filtered_df)

    st.subheader("🚨 Flagged Anomalies")
    flagged = filtered_df[(filtered_df['Anomaly'] == -1) & (filtered_df['Anomaly_Score'] < alert_threshold)]
    st.dataframe(flagged)

    st.download_button("Download Flagged Anomalies", flagged.to_csv(index=False), file_name="flagged_anomalies.csv")

    # Anomaly Score Distribution
    st.subheader("📊 Anomaly Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Anomaly_Score'], bins=30, kde=True, ax=ax)
    ax.axvline(alert_threshold, color='red', linestyle='--', label='Alert Threshold')
    ax.legend()
    st.pyplot(fig)

    # PCA Visualization
    st.subheader("📌 PCA Projection of Transactions")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)
    df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]
    fig2, ax2 = plt.subplots()
    colors = df['Anomaly'].map({1: 'blue', -1: 'red'})
    ax2.scatter(df['PCA1'], df['PCA2'], c=colors)
    ax2.set_xlabel("PCA 1")
    ax2.set_ylabel("PCA 2")
    ax2.set_title("PCA of Transactions")
    st.pyplot(fig2)
else:
    st.info("Upload a procurement Excel file to get started.")
