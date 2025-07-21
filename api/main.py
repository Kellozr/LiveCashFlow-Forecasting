from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from calendar import monthrange
import requests
from dateutil.relativedelta import relativedelta
try:
    from prophet import Prophet
    has_prophet = True
except ImportError:
    has_prophet = False
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
models_dir = os.path.join(root_dir, 'models')
data_dir = os.path.join(root_dir, 'data')
db_path = os.path.join(data_dir, 'database.db')

# Load models
try:
    clustering_model = joblib.load(os.path.join(models_dir, 'clustering_model.pkl'))
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    from sklearn.cluster import KMeans
    clustering_model = KMeans()

try:
    fraud_model = joblib.load(os.path.join(models_dir, 'fraud_model.pkl'))
    print("Fraud model loaded successfully!")
except Exception as e:
    print(f"Error loading fraud model: {e}")
    fraud_model = None

class Transaction(BaseModel):
    amount: float
    hour_of_day: int
    is_foreign: bool
    days_since_last_txn: int
    txn_category: str = None
    merchant_category: str = None
    direction: str = None
    location: str = None
    date: str = None

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"

def groq_explain(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful financial assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 256,
        "temperature": 0.7
    }
    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Groq explanation unavailable. (Reason: {str(e)})"

@app.get("/")
def home():
    return {"message": "Cashflow API is running!"}

@app.get("/account_summary/{account_no}")
def account_summary(account_no: str):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM transactions WHERE account_no = ?", conn, params=(account_no,))
        if df.empty:
            return {"error": f"No data found for account {account_no}"}
        num_txns = len(df)
        avg_txn = df['amount'].mean()
        most_freq_cat = df['txn_category'].mode()[0] if 'txn_category' in df and not df['txn_category'].isnull().all() else None
        last_txn_date = df['date'].max() if 'date' in df else None
        monthly_trend = df.groupby(df['date'].str[:7])['amount'].sum().to_dict() if 'date' in df else {}
        total_spend = df['amount'].sum()
        return {
            "num_transactions": int(num_txns),
            "avg_transaction": float(avg_txn),
            "most_frequent_category": most_freq_cat,
            "last_transaction_date": last_txn_date,
            "monthly_trend": monthly_trend,
            "total_spend": float(total_spend)
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

@app.post("/predict_spend/{account_no}")
def predict_spend(account_no: str, start_date: str = Body(None), end_date: str = Body(None), mode: str = Body("custom")):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT date, amount, txn_category FROM transactions WHERE account_no = ?", conn, params=(account_no,))
        if df.empty:
            return {"predicted_total": 0, "predicted_by_category": {}, "historical_total": 0, "historical_by_category": {}, "trend": {}, "error": "No data found for account.", "gemini_explanation": "No data available for explanation."}
        df['date'] = pd.to_datetime(df['date'])
        today = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
        if mode == "month_end":
            start = today
            end = today.replace(day=monthrange(today.year, today.month)[1])
        else:
            start = pd.to_datetime(start_date) if start_date else today
            end = pd.to_datetime(end_date) if end_date else today
        days = (end - start).days + 1
        if days <= 0:
            return {"predicted_total": 0, "predicted_by_category": {}, "historical_total": 0, "historical_by_category": {}, "trend": {}, "error": "Invalid date range.", "gemini_explanation": "Invalid date range."}
        # Calculate average daily spend per category for the last 3 months
        three_months_ago = start - pd.DateOffset(months=3)
        recent_df = df[df['date'] >= three_months_ago]
        avg_daily = recent_df.groupby('txn_category')['amount'].sum() / max(1, (start - three_months_ago).days)
        pred_by_cat = (avg_daily * days).to_dict()
        pred_total = sum(pred_by_cat.values())
        # Historical comparison: same period last year
        last_year_start = start - pd.DateOffset(years=1)
        last_year_end = end - pd.DateOffset(years=1)
        mask_last_year = (df['date'] >= last_year_start) & (df['date'] <= last_year_end)
        hist_by_cat = df[mask_last_year].groupby('txn_category')['amount'].sum().to_dict()
        hist_total = sum(hist_by_cat.values())
        # Trend: compare last 30 days to previous 30 days
        last_30 = df[df['date'] >= (today - pd.Timedelta(days=30))]
        prev_30 = df[(df['date'] < (today - pd.Timedelta(days=30))) & (df['date'] >= (today - pd.Timedelta(days=60)))]
        trend = {}
        for cat in set(last_30['txn_category']).union(prev_30['txn_category']):
            last = last_30[last_30['txn_category'] == cat]['amount'].sum()
            prev = prev_30[prev_30['txn_category'] == cat]['amount'].sum()
            if prev > 0:
                trend[cat] = round(100 * (last - prev) / prev, 1)
            else:
                trend[cat] = None
        # Groq explanation
        prompt = (
            f"A user asked for a spending prediction from {start.date()} to {end.date()} for their account.\n"
            f"Predicted total: ${pred_total:.2f}\n"
            f"Predicted by category: {pred_by_cat}\n"
            f"Historical total (same period last year): ${hist_total:.2f}\n"
            f"Historical by category: {hist_by_cat}\n"
            f"Trend (last 30 days vs previous 30): {trend}\n"
            f"Explain in plain English what this means, highlight which categories are driving the prediction, compare to last year, and suggest actions to stay within budget."
        )
        groq_explanation = groq_explain(prompt)
        return {
            "predicted_total": float(pred_total),
            "predicted_by_category": {k: float(v) for k, v in pred_by_cat.items()},
            "historical_total": float(hist_total),
            "historical_by_category": {k: float(v) for k, v in hist_by_cat.items()},
            "trend": trend,
            "error": None,
            "gemini_explanation": groq_explanation
        }
    except Exception as e:
        return {"predicted_total": 0, "predicted_by_category": {}, "historical_total": 0, "historical_by_category": {}, "trend": {}, "error": str(e), "gemini_explanation": f"Error: {str(e)}"}
    finally:
        conn.close()

@app.post("/predict_future_spend/{account_no}")
def predict_future_spend(account_no: str, months: int = Body(6)):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT date, amount, txn_category FROM transactions WHERE account_no = ?", conn, params=(account_no,))
        if df.empty:
            return {"predictions": [], "error": "No data found for account."}
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        # Group by month and category
        monthly_cat = df.groupby(['month', 'txn_category'])['amount'].sum().unstack(fill_value=0)
        monthly_total = monthly_cat.sum(axis=1)
        # Use the last N months for baseline prediction
        N = min(6, len(monthly_cat))
        avg_by_cat = monthly_cat.tail(N).mean()
        avg_total = monthly_total.tail(N).mean()
        # Forecast next 1-6 months
        last_month = monthly_cat.index.max().to_timestamp()
        predictions = []
        for i in range(1, months+1):
            future_month = (last_month + relativedelta(months=i)).strftime('%Y-%m')
            pred_by_cat = {cat: float(avg_by_cat[cat]) for cat in avg_by_cat.index}
            pred_total = float(sum(pred_by_cat.values()))
            predictions.append({
                "month": future_month,
                "predicted_total": pred_total,
                "predicted_by_category": pred_by_cat
            })
        return {"predictions": predictions, "error": None}
    except Exception as e:
        return {"predictions": [], "error": str(e)}
    finally:
        conn.close()

@app.get("/spend_categories/{account_no}")
def spend_categories(account_no: str):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT txn_category, amount FROM transactions WHERE account_no = ?", conn, params=(account_no,))
        if df.empty:
            return {"error": f"No data found for account {account_no}"}
        categories = df.groupby('txn_category')['amount'].sum().to_dict()
        return {"categories": categories}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

@app.get("/cost_cut_suggestions/{account_no}")
def cost_cut_suggestions(account_no: str):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT date, amount, txn_category FROM transactions WHERE account_no = ?", conn, params=(account_no,))
        if df.empty:
            return {
                "suggestions": [],
                "simulated_savings": {},
                "rising_categories": [],
                "trend": {},
                "error": "No data found for account.",
                "gemini_explanation": "No data available for explanation."
            }
        cat_sum = df.groupby('txn_category')['amount'].sum().sort_values(ascending=False)
        simulated = {cat: float(val * 0.2) for cat, val in cat_sum.head(2).items()}
        # Filter out negative or zero simulated savings
        simulated = {cat: val for cat, val in simulated.items() if val > 0}
        df['date'] = pd.to_datetime(df['date'])
        cutoff = df['date'].max() - pd.Timedelta(days=30)
        last_30 = df[df['date'] > cutoff]
        prev_30 = df[(df['date'] <= cutoff) & (df['date'] > cutoff - pd.Timedelta(days=30))]
        rising = []
        for cat in cat_sum.index:
            last = last_30[last_30['txn_category'] == cat]['amount'].sum()
            prev = prev_30[prev_30['txn_category'] == cat]['amount'].sum()
            if prev > 0 and last > prev * 1.2:
                rising.append(cat)
        trend = {}
        for cat in cat_sum.head(3).index:
            last = last_30[last_30['txn_category'] == cat]['amount'].sum()
            prev = prev_30[prev_30['txn_category'] == cat]['amount'].sum()
            if prev > 0:
                trend[cat] = round(100 * (last - prev) / prev, 1)
            else:
                trend[cat] = None
        suggestions = []
        for cat in simulated:
            suggestions.append(f"Reduce {cat} by 20% to save ${simulated[cat]:.2f} this month.")
        for cat in rising:
            if cat not in simulated:
                suggestions.append(f"Spending in {cat} is rising. Review recent expenses.")
        # Groq explanation
        prompt = f"A user received these cost-cutting suggestions: {suggestions}. Simulated savings: {simulated}. Rising categories: {rising}. Trend: {trend}. Only mention suggestions with positive savings. Explain in plain English why these suggestions were made, which categories are most important, and how the user can act on them. Ignore any suggestion with negative or zero savings."
        groq_explanation = groq_explain(prompt)
        return {
            "suggestions": suggestions,
            "simulated_savings": simulated,
            "rising_categories": rising,
            "trend": trend,
            "error": None,
            "gemini_explanation": groq_explanation
        }
    except Exception as e:
        return {
            "suggestions": [],
            "simulated_savings": {},
            "rising_categories": [],
            "trend": {},
            "error": str(e),
            "gemini_explanation": f"Error: {str(e)}"
        }
    finally:
        conn.close()

@app.get("/financial_health/{account_no}")
def financial_health(account_no: str):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT amount, txn_category FROM transactions WHERE account_no = ?", conn, params=(account_no,))
        if df.empty:
            return {"score": 0, "percentile": 0, "advice": ["No data found for this account."]}
        avg_spend = df['amount'].mean()
        total_spend = df['amount'].sum()
        # Compare to cluster
        cluster_df = pd.read_sql_query("SELECT account_no, AVG(amount) as avg_amount, SUM(amount) as total_spend FROM transactions GROUP BY account_no", conn)
        cluster_df['cluster'] = clustering_model.predict(cluster_df[['avg_amount', 'total_spend']])
        this_row = cluster_df[cluster_df['account_no'] == account_no]
        if this_row.empty:
            return {"score": 0, "percentile": 0, "advice": ["Account not found in cluster analysis."]}
        this_cluster = this_row['cluster'].values[0]
        cluster_members = cluster_df[cluster_df['cluster'] == this_cluster]
        percentile = (cluster_members['total_spend'] < total_spend).mean() * 100
        # Score: higher percentile = better (spends less than others)
        score = int(100 - percentile)
        advice = []
        if score > 80:
            advice.append("Excellent! You spend less than most users in your group.")
        elif score > 60:
            advice.append("Good. You are better than average, but review your top categories for more savings.")
        else:
            advice.append("You spend more than most users in your group. Consider reviewing your largest expense categories.")
        # Add top category advice
        top_cat = df.groupby('txn_category')['amount'].sum().sort_values(ascending=False).head(1)
        if not top_cat.empty:
            advice.append(f"Your top spending category is {top_cat.index[0]}. Try to reduce it by 10% for a higher score.")
        return {"score": score, "percentile": percentile, "advice": advice}
    except Exception as e:
        return {"score": 0, "percentile": 0, "advice": [f"Error: {str(e)}"]}
    finally:
        conn.close()

@app.get("/cluster/{account_no}")
def get_cluster(account_no: str):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        # Get account data
        query = f"SELECT * FROM transactions WHERE account_no = ?"
        df = pd.read_sql_query(query, conn, params=(account_no,))
        if df.empty:
            return {"error": f"No data found for account {account_no}"}
        avg_amount = df['amount'].mean()
        total_spend = df['amount'].sum()
        num_txns = len(df)
        avg_txn = avg_amount
        most_freq_cat = df['txn_category'].mode()[0] if 'txn_category' in df and not df['txn_category'].isna().all() else None
        last_txn_date = df['date'].max() if 'date' in df else None
        # Monthly trend (last 6 months)
        df['date'] = pd.to_datetime(df['date'])
        monthly_trend = df.groupby(df['date'].dt.to_period('M'))['amount'].sum().sort_index()
        monthly_trend = {str(period): float(val) for period, val in monthly_trend.tail(6).items()}
        # Predict cluster
        features = [[avg_amount, total_spend]]
        cluster = clustering_model.predict(features)[0]
        # Cluster center and comparison
        centers = clustering_model.cluster_centers_ if hasattr(clustering_model, 'cluster_centers_') else []
        cluster_center = centers[cluster].tolist() if len(centers) > cluster else []
        all_centers = [c.tolist() for c in centers] if len(centers) > 0 else []
        types = {0: "Saver", 1: "Balanced", 2: "Spender"}
        description = {
            0: "Savers spend less and have lower average transaction amounts.",
            1: "Balanced users have moderate spending and transaction sizes.",
            2: "Spenders have high total and average spend."
        }.get(cluster, "Unknown")
        return {
            "account": account_no,
            "cluster": int(cluster),
            "type": types.get(cluster, "Unknown"),
            "avg_amount": float(avg_amount),
            "total_spend": float(total_spend),
            "num_transactions": int(num_txns),
            "avg_transaction": float(avg_txn),
            "most_frequent_category": most_freq_cat,
            "last_transaction_date": str(last_txn_date) if last_txn_date else None,
            "monthly_trend": monthly_trend,
            "cluster_center": cluster_center,
            "all_centers": all_centers,
            "description": description
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn: conn.close()

@app.post("/gemini_chat")
def gemini_chat(question: str = Body(...), account_no: str = Body(None)):
    # Use Groq for chat
    context = ""
    if account_no:
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query("SELECT date, amount, txn_category FROM transactions WHERE account_no = ? ORDER BY date DESC LIMIT 10", conn, params=(account_no,))
            if not df.empty:
                context = f"Recent transactions for account {account_no}:\n" + df.to_string(index=False)
            conn.close()
        except Exception as e:
            context = "(Could not fetch account context)"
    prompt = f"You are a financial assistant. {context}\nUser question: {question}"
    answer = groq_explain(prompt)
    return {"answer": answer}

@app.get("/detect_fraud/{account_no}")
def detect_fraud(account_no: str):
    if fraud_model is None:
        return JSONResponse(status_code=500, content={"error": "Fraud model not available."})
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM transactions WHERE account_no = ?", conn, params=(account_no,))
        if df.empty:
            return {"error": f"No data found for account {account_no}"}
        features = ['amount', 'hour_of_day', 'is_foreign', 'days_since_last_txn']
        X = df[features].copy()
        # Ensure correct dtypes
        X['is_foreign'] = X['is_foreign'].astype(float)
        # Predict anomaly scores
        scores = fraud_model.decision_function(X)
        preds = fraud_model.predict(X)
        df['fraud_score'] = -scores  # Higher = more anomalous
        df['is_fraud'] = (preds == -1).astype(int)
        # Return flagged transactions and all with scores
        flagged = df[df['is_fraud'] == 1].to_dict(orient='records')
        all_txns = df.to_dict(orient='records')
        return {"flagged": flagged, "all": all_txns}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

class SpendingForecastRequest(BaseModel):
    account_no: str
    forecast_months: int = 3

@app.post("/spending_forecast")
def spending_forecast(request: SpendingForecastRequest):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        query = f"""
            SELECT strftime('%Y-%m', date) as month, 
                   txn_category, 
                   SUM(amount) as total_spent
            FROM transactions
            WHERE account_no = '{request.account_no}'
            GROUP BY strftime('%Y-%m', date), txn_category
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn)
        if df.empty:
            return {"error": "No transaction history found for this account"}
        current_date = datetime.now()
        forecast_data = {}
        all_months = []
        # Build per-category forecasts
        for category in df['txn_category'].unique():
            cat_df = df[df['txn_category'] == category].copy()
            min_date = datetime.strptime(cat_df['month'].min() + '-01', '%Y-%m-%d')
            max_date = datetime.strptime(cat_df['month'].max() + '-01', '%Y-%m-%d')
            months_range = pd.date_range(start=min_date, end=max_date, freq='MS')
            all_months = list(months_range)
            full_df = pd.DataFrame({'month': months_range.strftime('%Y-%m')})
            cat_df = pd.merge(full_df, cat_df, on='month', how='left')
            cat_df['total_spent'].fillna(0, inplace=True)
            predictions = []
            if has_prophet and len(cat_df) >= 6:
                prophet_df = cat_df.copy()
                prophet_df['ds'] = pd.to_datetime(prophet_df['month'] + '-01')
                prophet_df['y'] = prophet_df['total_spent']
                model = Prophet(weekly_seasonality=False, daily_seasonality=False)
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                model.fit(prophet_df[['ds', 'y']])
                future = model.make_future_dataframe(periods=request.forecast_months, freq='MS', include_history=False)
                forecast = model.predict(future)
                for _, row in forecast.iterrows():
                    predictions.append({
                        "month": row['ds'].strftime("%Y-%m"),
                        "predicted_amount": round(max(row['yhat'], 0), 2),
                        "category": category
                    })
            else:
                weights = [0.1, 0.2, 0.3, 0.4]
                last_n = min(4, len(cat_df))
                avg_spend = sum(cat_df['total_spent'].iloc[-last_n:] * weights[:last_n]) / sum(weights[:last_n])
                if len(cat_df) >= 12:
                    current_month = current_date.month
                    monthly_factors = []
                    for i in range(1, 13):
                        month_data = cat_df[cat_df['month'].str.endswith(f'-{i:02d}')]
                        if not month_data.empty:
                            monthly_factors.append(month_data['total_spent'].mean())
                    if monthly_factors:
                        seasonality_factor = monthly_factors[current_month-1] / max(sum(monthly_factors)/12, 1)
                        avg_spend *= seasonality_factor
                for i in range(1, request.forecast_months + 1):
                    forecast_date = current_date + relativedelta(months=i)
                    predictions.append({
                        "month": forecast_date.strftime("%Y-%m"),
                        "predicted_amount": round(avg_spend, 2),
                        "category": category
                    })
            forecast_data[category] = predictions
        # Combine into frontend format
        months = [ (current_date + relativedelta(months=i)).strftime("%Y-%m") for i in range(1, request.forecast_months+1) ]
        categories = list(forecast_data.keys())
        forecast = []
        for i, month in enumerate(months):
            row = {"month": month}
            total = 0
            for cat in categories:
                amt = forecast_data[cat][i]["predicted_amount"] if i < len(forecast_data[cat]) else 0
                row[cat] = amt
                total += amt
            row["total"] = total
            forecast.append(row)
        response = {
            "account": request.account_no,
            "forecast_months": request.forecast_months,
            "categories": categories,
            "forecast": forecast
        }
        return response
    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()

# For direct run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)