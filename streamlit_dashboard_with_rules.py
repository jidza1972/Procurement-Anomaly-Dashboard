
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Procurement Anomaly Dashboard", layout="wide")
st.title("📊 Procurement Fraud & Anomaly Detection Dashboard")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Procurement Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    st.sidebar.header("🔍 Filter Options")
    department_filter = st.sidebar.multiselect("Department", df['Department'].dropna().unique(), default=df['Department'].dropna().unique())
    method_filter = st.sidebar.multiselect("Procurement Method", df['Procurement_Method'].dropna().unique(), default=df['Procurement_Method'].dropna().unique())

    filtered_df = df[df['Department'].isin(department_filter) & df['Procurement_Method'].isin(method_filter)]

    st.subheader("📋 Filtered Transactions")
    st.dataframe(filtered_df)

    st.subheader("🚨 Rule-Based Anomalies")
    if 'Anomaly_BusinessRule' in df.columns:
        rule_anomalies = filtered_df[filtered_df['Anomaly_BusinessRule'] == 1]
        st.dataframe(rule_anomalies)
        st.download_button("Download Rule-Based Anomalies", rule_anomalies.to_csv(index=False), file_name="rule_based_anomalies.csv")
    else:
        st.warning("No 'Anomaly_BusinessRule' column found in dataset.")

    st.subheader("🧠 Model-Based Anomalies (Isolation Forest)")
    # Prepare features
    df_model = df.copy()
    non_features = ['Transaction_ID', 'Supplier_Name', 'Date', 'Date_Awarded', 'Date_Invoiced', 'Invoice_Description', 
                    'Anomaly_RFQ_OverThreshold', 'Anomaly_Tender_Splitting', 'Anomaly_BusinessRule']
    df_model = df_model.drop(columns=[col for col in non_features if col in df_model.columns], errors='ignore')
    df_model = df_model.fillna(0)

    # Encode categoricals
    cat_cols = df_model.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))

    # Scale numerical features
    numeric_cols = df_model.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_model[numeric_cols])

    # Apply Isolation Forest
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(df_scaled)
    df['Model_Anomaly_Score'] = model.decision_function(df_scaled)
    df['Model_Anomaly'] = model.predict(df_scaled)
    df['Model_Anomaly_Label'] = df['Model_Anomaly'].map({1: 'Normal', -1: 'Anomaly'})

    # Show anomalies
    model_anomalies = df[(df['Model_Anomaly'] == -1)]
    st.dataframe(model_anomalies)
    st.download_button("Download Model-Based Anomalies", model_anomalies.to_csv(index=False), file_name="model_based_anomalies.csv")

    # Visualizations
    st.subheader("📌 PCA Visualization of Model-Based Anomalies")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)
    df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]
    fig, ax = plt.subplots()
    colors = df['Model_Anomaly'].map({1: 'blue', -1: 'red'})
    ax.scatter(df['PCA1'], df['PCA2'], c=colors)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("PCA of Procurement Transactions")
    st.pyplot(fig)
else:
    st.info("Upload an Excel procurement file to begin.")
