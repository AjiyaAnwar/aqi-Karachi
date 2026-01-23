"""
Time-series models without Prophet (for when Prophet fails)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

def get_historical_data():
    """Get historical AQI data"""
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE')]
    
    cursor = db.aqi_features.find({}, {'_id': 0, 'timestamp': 1, 'aqi': 1})
    df = pd.DataFrame(list(cursor))
    client.close()
    
    if df.empty:
        return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    return df

def simple_seasonal_forecast(df, periods=72):
    """Simple seasonal forecast based on daily patterns"""
    print("🤖 Creating seasonal forecast...")
    
    # Extract hour of day
    df['hour'] = df['timestamp'].dt.hour
    
    # Calculate average by hour
    hourly_avg = df.groupby('hour')['aqi'].mean()
    
    # Create forecast
    last_time = df['timestamp'].iloc[-1]
    predictions = []
    
    for i in range(1, periods + 1):
        forecast_time = last_time + timedelta(hours=i)
        hour = forecast_time.hour
        day_of_week = forecast_time.weekday()
        
        # Base prediction from hourly average
        base_pred = hourly_avg.get(hour, df['aqi'].mean())
        
        # Adjust for weekday/weekend
        if day_of_week >= 5:  # Weekend
            base_pred *= 0.95  # Slightly lower on weekends
        
        # Add small random variation
        variation = np.random.normal(0, 5)
        predicted = max(10, base_pred + variation)  # AQI can't be negative
        
        predictions.append({
            'timestamp': forecast_time,
            'predicted_aqi': predicted,
            'model': 'seasonal'
        })
    
    return pd.DataFrame(predictions)

def exponential_smoothing_forecast(df, periods=72, alpha=0.3):
    """Exponential smoothing forecast"""
    print("🤖 Creating exponential smoothing forecast...")
    
    # Simple exponential smoothing
    aqi_series = df['aqi'].values
    last_value = aqi_series[-1]
    
    predictions = []
    forecast_time = df['timestamp'].iloc[-1]
    
    for i in range(1, periods + 1):
        # Simple exponential smoothing
        if i == 1:
            predicted = last_value
        else:
            # Use weighted average of recent values
            weights = np.exp(-alpha * np.arange(1, min(25, len(aqi_series))))
            weights = weights / weights.sum()
            recent_values = aqi_series[-len(weights):]
            predicted = np.sum(recent_values * weights)
        
        # Add time of day effect
        hour = (forecast_time + timedelta(hours=i)).hour
        if 8 <= hour <= 20:
            predicted *= 1.05  # 5% higher during day
        
        predictions.append({
            'timestamp': forecast_time + timedelta(hours=i),
            'predicted_aqi': predicted,
            'model': 'exponential_smoothing'
        })
    
    return pd.DataFrame(predictions)

def save_to_mongodb(predictions_df, collection_name):
    """Save predictions to MongoDB"""
    if predictions_df.empty:
        return
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE')]
    
    # Clear old predictions
    db[collection_name].delete_many({})
    
    # Prepare records
    records = predictions_df.to_dict('records')
    for record in records:
        record['timestamp'] = record['timestamp'].isoformat()
        record['created_at'] = datetime.now().isoformat()
    
    # Save
    if records:
        db[collection_name].insert_many(records)
        print(f"💾 Saved {len(records)} predictions to {collection_name}")
    
    client.close()

def main():
    """Main without Prophet"""
    print("=" * 60)
    print("📈 SIMPLE TIME-SERIES FORECAST (No Prophet)")
    print("=" * 60)
    
    # Load data
    df = get_historical_data()
    if df is None:
        print("❌ No data available")
        return
    
    print(f"📊 Loaded {len(df)} historical records")
    print(f"📅 From {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"📈 Current AQI: {df['aqi'].iloc[-1]:.1f}")
    
    # Generate forecasts
    print("\n" + "=" * 60)
    
    # 1. Seasonal forecast
    seasonal_pred = simple_seasonal_forecast(df)
    save_to_mongodb(seasonal_pred, 'simple_forecasts')
    
    # 2. Exponential smoothing
    es_pred = exponential_smoothing_forecast(df)
    save_to_mongodb(es_pred, 'simple_forecasts')
    
    # 3. Moving average (simple)
    print("\n🤖 Creating moving average forecast...")
    ma_24h = df['aqi'].tail(24).mean()
    ma_7d = df['aqi'].tail(168).mean() if len(df) >= 168 else df['aqi'].mean()
    
    ma_preds = []
    last_time = df['timestamp'].iloc[-1]
    
    for i in range(1, 73):
        forecast_time = last_time + timedelta(hours=i)
        
        # Weighted average of moving averages
        predicted = ma_24h * 0.7 + ma_7d * 0.3
        
        # Daily pattern
        hour = forecast_time.hour
        if 8 <= hour <= 20:
            predicted *= 1.08
        
        ma_preds.append({
            'timestamp': forecast_time,
            'predicted_aqi': predicted,
            'model': 'moving_average'
        })
    
    ma_df = pd.DataFrame(ma_preds)
    save_to_mongodb(ma_df, 'simple_forecasts')
    
    # Show predictions
    print("\n" + "=" * 60)
    print("📅 3-DAY FORECAST SUMMARY")
    print("=" * 60)
    
    # Combine all predictions
    all_preds = pd.concat([seasonal_pred, es_pred, ma_df], ignore_index=True)
    
    # Group by date and show average
    all_preds['timestamp'] = pd.to_datetime(all_preds['timestamp'])
    all_preds['date'] = all_preds['timestamp'].dt.date
    
    daily_avg = all_preds.groupby('date').agg({
        'predicted_aqi': 'mean'
    }).reset_index()
    
    for _, row in daily_avg.head(3).iterrows():
        print(f"  {row['date']}: AQI {row['predicted_aqi']:.1f}")
    
    # Save ensemble
    ensemble_records = daily_avg.to_dict('records')
    for record in ensemble_records:
        record['date'] = record['date'].isoformat()
        record['created_at'] = datetime.now().isoformat()
        record['model'] = 'ensemble'
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE')]
    db.ensemble_predictions.delete_many({})
    db.ensemble_predictions.insert_many(ensemble_records)
    client.close()
    
    print(f"\n💾 Saved ensemble forecast for {len(ensemble_records)} days")
    print(f"\n✅ Done! Run dashboard with: streamlit run app/time_series_dashboard.py")

if __name__ == "__main__":
    main()
 