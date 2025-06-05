# Procurement Anomaly Detection Dashboard

This dashboard identifies anomalies in procurement data using:
1. ✅ Rule-based business logic (e.g. RFQ > $10,000, tender splitting)
2. ✅ Machine learning (Isolation Forest algorithm)

## 📦 Contents
- `procurement_data_with_rule_anomalies.xlsx` – Enhanced dataset with anomaly flags
- `streamlit_dashboard_with_rules.py` – Final Streamlit dashboard
- `requirements.txt` – Python libraries needed
- `README.txt` – Instructions

## ▶️ How to Run Locally
1. Install required libraries:
   pip install -r requirements.txt

2. Run the app:
   streamlit run streamlit_dashboard_with_rules.py

3. Upload the Excel dataset in the app interface and explore anomalies.

## ☁️ Deploy to Streamlit Cloud
1. Push all files to a public GitHub repository
2. Go to https://streamlit.io/cloud
3. Connect your GitHub and set `streamlit_dashboard_with_rules.py` as the main file
4. Click "Deploy"

This tool supports both internal audits and proactive fraud detection in public procurement.