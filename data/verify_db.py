# data/verify_db.py
import sqlite3
import pandas as pd
import os

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, 'database.db')

print(f"Database path: {db_path}")
print(f"Database exists: {os.path.exists(db_path)}")

# Connect to database
conn = sqlite3.connect(db_path)

# Check tables
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("\nTables in database:")
for table in tables:
    print(table[0])

# Check transactions table
if 'transactions' in [t[0] for t in tables]:
    df = pd.read_sql_query("SELECT * FROM transactions LIMIT 5", conn)
    print("\nFirst 5 transactions:")
    print(df)
    print(f"\nTotal records: {len(pd.read_sql_query('SELECT * FROM transactions', conn))}")

conn.close()