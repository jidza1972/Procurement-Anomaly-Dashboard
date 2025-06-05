# Procurement Fraud Anomaly Detection Dashboard

This project implements an Isolation Forest-based anomaly detection system tailored for public procurement transactions, such as those at POTRAZ.

## Contents
- `procurement_data_mock.xlsx`: Sample mock data
- `streamlit_procurement_dashboard_v2.py`: Final dashboard app
- `requirements.txt`: Python dependencies
- `README.txt`: Instructions

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the app:
   streamlit run streamlit_procurement_dashboard_v2.py

3. Upload the Excel file and explore the results.

## Deploy to Streamlit Cloud

1. Push all files to a GitHub repo
2. Go to https://streamlit.io/cloud
3. Select your repo and set main file as `streamlit_procurement_dashboard_v2.py`

## Output

- Detects procurement anomalies
- Allows filtering and downloads
- Visualizes anomalies via PCA

Created for risk-based audit and fraud detection.