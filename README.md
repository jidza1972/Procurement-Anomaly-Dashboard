
# Procurement Fraud Anomaly Detection Dashboard

This project implements an Isolation Forest-based anomaly detection system tailored for public procurement transactions, such as those at POTRAZ.

## 📦 Contents
- `procurement_data_mock.xlsx`: Sample mock data for procurement transactions
- `streamlit_procurement_dashboard.py`: Streamlit app for interactive anomaly detection
- `README.txt`: This instruction file

## 🛠️ Requirements

Install required Python libraries using pip:

```
pip install streamlit pandas scikit-learn openpyxl matplotlib seaborn
```

## ▶️ Running the Dashboard Locally

1. Ensure Python is installed.
2. Place `procurement_data_mock.xlsx` and `streamlit_procurement_dashboard.py` in the same folder.
3. Launch the app using the command:

```
streamlit run streamlit_procurement_dashboard.py
```

4. Upload the Excel file using the file uploader widget in the app.

## 🚨 What It Does

- Detects suspicious procurement transactions using Isolation Forest
- Flags transactions with low anomaly scores
- Visualizes results with PCA and score distribution plots
- Allows filtering by department and procurement method
- Enables download of flagged anomalies for audit review

## ☁️ Deploying on Streamlit Cloud

1. Create a free GitHub account if you don't have one.
2. Push these files to a GitHub repository.
3. Go to https://streamlit.io/cloud and connect your GitHub.
4. Deploy your app by selecting your repository and setting the main script to `streamlit_procurement_dashboard.py`.

## 🏢 Internal Deployment (Optional)

- Ensure Python environment is installed on your internal server.
- Use a process manager like `pm2` or `systemd` to run the app persistently.
- Set firewall rules or use Nginx for access control.

## 📧 Support

For guidance, consult your internal audit or IT support team. You can also contact the developer of this tool for configuration help.

---

Created with ❤️ for fraud risk management in public entities.
