# data/initialize_db.py
import pandas as pd
import sqlite3
from pathlib import Path

# Setup paths
data_dir = Path(__file__).parent
csv_path = data_dir / 'ring_fraud_data.csv'
db_path = data_dir / 'database.db'

# Load CSV
df = pd.read_csv(csv_path)

# Connect to SQLite
conn = sqlite3.connect(db_path)

# Save to database
df.to_sql('transactions', conn, if_exists='replace', index=False)

print(f"Database initialized with {len(df)} records at {db_path}")