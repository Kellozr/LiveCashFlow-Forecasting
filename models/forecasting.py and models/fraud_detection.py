# models/fraud_detection.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import joblib
import shap
from sqlalchemy import create_engine

class LSTMFraudDetector:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.model = self._build_model()
        self.explainer = None
    
    def _build_model(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(self.sequence_length, 15), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(32))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def train(self, db_path='data/transactions.db'):
        engine = create_engine(f'sqlite:///{db_path}')
        query = """
            SELECT account_no, amount, balance_amt, day_of_week, is_weekend, 
                   hour_of_day, month, amount_category, is_foreign, days_since_last_txn,
                   txn_category, merchant_category, direction, location, is_fraud
            FROM transactions
        """
        df = pd.read_sql_query(query, engine)
        
        # Preprocess
        categorical_cols = ['txn_category', 'merchant_category', 'direction', 'location']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Scale numerical features
        numerical_cols = ['amount', 'balance_amt', 'day_of_week', 'hour_of_day', 
                          'month', 'days_since_last_txn']
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        # Create sequences
        X, y = [], []
        accounts = df['account_no'].unique()
        
        for account in accounts:
            account_data = df[df['account_no'] == account].sort_values('date')
            features = account_data.drop(['account_no', 'is_fraud'], axis=1).values
            fraud_labels = account_data['is_fraud'].values
            
            for i in range(len(account_data) - self.sequence_length):
                X.append(features[i:i+self.sequence_length])
                y.append(fraud_labels[i+self.sequence_length])
        
        X, y = np.array(X), np.array(y)
        
        # Train model
        self.model.fit(X, y, epochs=20, batch_size=64, validation_split=0.2, verbose=0)
        
        # Create SHAP explainer
        background = X[np.random.choice(X.shape[0], 100, replace=False)]
        self.explainer = shap.DeepExplainer(self.model, background)
        
        # Save model
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'explainer': self.explainer,
            'feature_names': list(df.drop(['account_no', 'is_fraud'], axis=1).columns)
        }, 'models/fraud_model.pkl')
    
    def predict(self, transaction_history):
        model_data = joblib.load('models/fraud_model.pkl')
        model = model_data['model']
        scaler = model_data['scaler']
        explainer = model_data['explainer']
        feature_names = model_data['feature_names']
        
        # Preprocess transaction
        transaction_df = pd.DataFrame([transaction_history])
        transaction_df = pd.get_dummies(transaction_df)
        
        # Ensure all columns are present
        for col in feature_names:
            if col not in transaction_df.columns:
                transaction_df[col] = 0
        
        transaction_df = transaction_df[feature_names]
        
        # Scale numerical features
        numerical_cols = ['amount', 'balance_amt', 'day_of_week', 'hour_of_day', 
                          'month', 'days_since_last_txn']
        transaction_df[numerical_cols] = scaler.transform(transaction_df[numerical_cols])
        
        # Predict
        probability = model.predict(transaction_df.values.reshape(1, 1, -1))[0][0]
        is_fraud = 1 if probability > 0.7 else 0
        
        # Explain
        shap_values = explainer.shap_values(transaction_df.values.reshape(1, 1, -1))
        
        return {
            'is_fraud': is_fraud,
            'probability': float(probability),
            'shap_values': shap_values[0][0].tolist(),
            'feature_names': feature_names,
            'feature_values': transaction_df.values[0].tolist()
        }