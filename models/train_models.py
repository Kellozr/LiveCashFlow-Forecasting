import os
import sqlite3  # Import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from prophet import Prophet
from xgboost import XGBRegressor

# Get absolute path to data directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')
db_path = os.path.join(data_dir, 'database.db')

print(f"Connecting to database at: {db_path}")

# Connect to SQLite database
conn = sqlite3.connect(db_path)

# Load data from database
try:
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    print(f"Loaded {len(df)} records from database")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Close connection
conn.close()

# Basic preprocessing
print("Preprocessing data...")
df['date'] = pd.to_datetime(df['date'])
df = df.dropna(subset=['account_no', 'date', 'amount', 'balance_amt'])

# 1. Forecasting Model (Prophet)
print("Training forecasting models...")
accounts = df['account_no'].unique()[:3]  # Limit to 3 accounts for demo

for account in accounts:
    try:
        account_df = df[df['account_no'] == account][['date', 'balance_amt']]
        account_df.columns = ['ds', 'y']
        if len(account_df) > 10:  # Need sufficient data
            model = Prophet()
            model.fit(account_df)
            model_path = os.path.join(current_dir, f'forecast_{account}.pkl')
            joblib.dump(model, model_path)
            print(f"Saved forecast model for account: {account}")
    except Exception as e:
        print(f"Error training forecast for {account}: {e}")

# 2. Fraud Detection Model
print("Training fraud detection model...")
try:
    fraud_features = ['amount', 'hour_of_day', 'is_foreign', 'days_since_last_txn']
    fraud_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', IsolationForest(contamination=0.01, random_state=42))
    ])
    fraud_pipeline.fit(df[fraud_features])
    joblib.dump(fraud_pipeline, os.path.join(current_dir, 'fraud_model.pkl'))
    print("Saved fraud detection model")
except Exception as e:
    print(f"Error training fraud model: {e}")

# 3. Clustering Model
print("Training clustering model...")
try:
    spending_features = df.groupby('account_no').agg({
        'amount': ['mean', 'sum'],
        'is_weekend': 'mean'
    }).reset_index()
    spending_features.columns = ['account_no', 'avg_amount', 'total_spend', 'weekend_ratio']
    
    cluster_model = KMeans(n_clusters=3, random_state=42)
    cluster_model.fit(spending_features[['avg_amount', 'total_spend']])
    joblib.dump(cluster_model, os.path.join(current_dir, 'clustering_model.pkl'))
    print("Saved clustering model")
except Exception as e:
    print(f"Error training clustering model: {e}")

print("Model training completed!")