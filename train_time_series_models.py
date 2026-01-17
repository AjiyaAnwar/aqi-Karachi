"""
Specialized time-series models for AQI prediction
Using Prophet and LSTM
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Time-series specific models
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

def prepare_prophet_data(df):
    """Prepare data for Facebook Prophet"""
    prophet_df = df[['timestamp', 'aqi']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df = prophet_df.sort_values('ds')
    
    # Add additional regressors
    prophet_df['pm25'] = df['pm25'].values
    prophet_df['pm10'] = df['pm10'].values
    prophet_df['hour'] = df['hour'].values
    
    return prophet_df

def train_prophet_model(df, forecast_hours=72):
    """Train Facebook Prophet model"""
    print("🤖 Training Prophet model...")
    
    # Prepare data
    prophet_df = prepare_prophet_data(df)
    
    # Initialize model with regressors
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='additive'
    )
    
    # Add regressors
    model.add_regressor('pm25')
    model.add_regressor('pm10')
    model.add_regressor('hour')
    
    # Fit model
    model.fit(prophet_df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_hours, freq='H')
    
    # Add regressors to future (using last known values for simplicity)
    future['pm25'] = prophet_df['pm25'].iloc[-1]
    future['pm10'] = prophet_df['pm10'].iloc[-1]
    future['hour'] = future['ds'].dt.hour
    
    # Make forecast
    forecast = model.predict(future)
    
    # Extract predictions
    predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_hours)
    predictions.columns = ['timestamp', 'predicted_aqi', 'lower_bound', 'upper_bound']
    
    print(f"✅ Prophet trained - made {forecast_hours}h forecast")
    return model, predictions

def prepare_lstm_data(df, sequence_length=24):
    """Prepare data for LSTM"""
    # Use only AQI for LSTM (univariate)
    data = df['aqi'].values.reshape(-1, 1)
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape for LSTM [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Split data (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler

def train_lstm_model(X_train, y_train, X_test, y_test):
    """Train LSTM model"""
    print("🤖 Training LSTM model...")
    
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Inverse transform predictions
    from sklearn.metrics import r2_score, mean_absolute_error
    
    # For simplicity, just return model and history
    print("✅ LSTM trained")
    return model, history

def main():
    """Time-series focused training"""
    print("⏰ TIME-SERIES AQI PREDICTION MODELS")
    print("=" * 60)
    
    # Load data (using existing features)
    from pymongo import MongoClient
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE')]
    
    cursor = db.aqi_features.find({}, {'_id': 0, 'created_at': 0})
    df = pd.DataFrame(list(cursor))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    print(f"📊 Loaded {len(df)} records")
    
    # 1. Train Prophet
    print("\n" + "=" * 60)
    print("1. FACEBOOK PROPHET MODEL")
    print("=" * 60)
    
    prophet_model, prophet_predictions = train_prophet_model(df)
    
    print(f"\n📅 Prophet 3-day forecast:")
    for _, row in prophet_predictions.head(24).iterrows():
        print(f"  {row['timestamp']}: AQI {row['predicted_aqi']:.1f}")
    
    # Save Prophet predictions
    os.makedirs('results', exist_ok=True)
    prophet_predictions.to_csv('results/prophet_forecast.csv', index=False)
    
    # 2. Train LSTM
    print("\n" + "=" * 60)
    print("2. LSTM MODEL")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df)
    lstm_model, history = train_lstm_model(X_train, y_train, X_test, y_test)
    
    # Save LSTM model
    import pickle
    os.makedirs('models', exist_ok=True)
    lstm_model.save('models/lstm_aqi_model.h5')
    
    with open('models/lstm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # 3. Simple Baseline: Moving Average
    print("\n" + "=" * 60)
    print("3. MOVING AVERAGE BASELINE")
    print("=" * 60)
    
    # Simple 7-day moving average forecast
    last_7_days_avg = df['aqi'].tail(168).mean()  # 7 days * 24 hours
    
    print(f"📊 Last 7-day average AQI: {last_7_days_avg:.1f}")
    
    # Create simple forecast
    forecast_dates = pd.date_range(
        start=df['timestamp'].max() + timedelta(hours=1),
        periods=72,
        freq='H'
    )
    
    baseline_forecast = pd.DataFrame({
        'timestamp': forecast_dates,
        'predicted_aqi': last_7_days_avg
    })
    
    # Add some variation based on hour of day
    for i, row in baseline_forecast.iterrows():
        hour = row['timestamp'].hour
        # Simple pattern: higher during day, lower at night
        if 6 <= hour <= 18:  # Daytime
            baseline_forecast.at[i, 'predicted_aqi'] = last_7_days_avg * 1.1
        else:  # Nighttime
            baseline_forecast.at[i, 'predicted_aqi'] = last_7_days_avg * 0.9
    
    baseline_forecast.to_csv('results/baseline_forecast.csv', index=False)
    
    # 4. Compare approaches
    print("\n" + "=" * 60)
    print("📊 FORECAST COMPARISON")
    print("=" * 60)
    
    print("\n📅 Tomorrow's AQI (24h ahead):")
    print(f"  Prophet:      {prophet_predictions.iloc[23]['predicted_aqi']:.1f}")
    print(f"  Baseline MA:  {baseline_forecast.iloc[23]['predicted_aqi']:.1f}")
    
    # Save all predictions to MongoDB
    from pymongo import MongoClient
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE')]
    
    # Save Prophet predictions
    prophet_records = prophet_predictions.to_dict('records')
    for record in prophet_records:
        record['model'] = 'prophet'
        record['created_at'] = datetime.now().isoformat()
        record['timestamp'] = record['timestamp'].isoformat()
    
    # Save baseline predictions
    baseline_records = baseline_forecast.to_dict('records')
    for record in baseline_records:
        record['model'] = 'moving_average'
        record['created_at'] = datetime.now().isoformat()
        record['timestamp'] = record['timestamp'].isoformat()
    
    # Clear and save
    db.time_series_predictions.delete_many({})
    db.time_series_predictions.insert_many(prophet_records + baseline_records)
    
    print(f"\n💾 Saved {len(prophet_records) + len(baseline_records)} predictions to MongoDB")
    
    client.close()
    
    print("\n" + "=" * 60)
    print("✅ TIME-SERIES MODELS TRAINED")
    print("=" * 60)
    print(f"📁 Files saved:")
    print(f"  results/prophet_forecast.csv")
    print(f"  results/baseline_forecast.csv")
    print(f"  models/lstm_aqi_model.h5")
    print(f"  models/lstm_scaler.pkl")

if __name__ == "__main__":
    main()
