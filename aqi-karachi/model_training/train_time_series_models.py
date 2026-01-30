"""
FIXED Time-series models without Prophet (for when Prophet fails)
UPDATED: Saves to Model Registry
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
from dotenv import load_dotenv
from pymongo import MongoClient
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

def get_historical_data():
    """Get historical AQI data"""
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE', 'aqi_predictor')]
    
    cursor = db.aqi_measurements.find({}, {'_id': 0, 'timestamp': 1, 'aqi': 1})
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
        variation = np.random.normal(0, 3)
        predicted = max(10, base_pred + variation)  # AQI can't be negative
        
        predictions.append({
            'timestamp': forecast_time,
            'predicted_aqi': predicted,
            'model': 'seasonal',
            'hour': hour,
            'day_of_week': day_of_week
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
            'model': 'exponential_smoothing',
            'hour': hour
        })
    
    return pd.DataFrame(predictions)

def save_to_mongodb(predictions_df, collection_name):
    """Save predictions to MongoDB"""
    if predictions_df.empty:
        return
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE', 'aqi_predictor')]
    
    # Clear old predictions
    db[collection_name].delete_many({})
    
    # Prepare records
    records = predictions_df.to_dict('records')
    for record in records:
        if 'timestamp' in record and isinstance(record['timestamp'], pd.Timestamp):
            record['timestamp'] = record['timestamp'].isoformat()
        record['created_at'] = datetime.now().isoformat()
    
    # Save
    if records:
        db[collection_name].insert_many(records)
        print(f"💾 Saved {len(records)} predictions to {collection_name}")
    
    client.close()

def save_time_series_to_registry(model_type, metrics, forecasts_count):
    """Save time series model to Model Registry"""
    try:
        client = MongoClient(os.getenv('MONGODB_URI'))
        registry_db = os.getenv('MODEL_REGISTRY_DB', 'aqi_model_registry')
        db = client[registry_db]
        
        # Get feature version
        feature_store_db = os.getenv('FEATURE_STORE_DB', 'aqi_feature_store')
        fs_db = client[feature_store_db]
        
        feature_version = "unknown"
        if 'feature_versions' in fs_db.list_collection_names():
            latest_version = fs_db['feature_versions'].find_one(sort=[('timestamp', -1)])
            if latest_version:
                feature_version = latest_version.get('version', 'unknown')
        
        # Create collection if not exists
        if 'time_series_models' not in db.list_collection_names():
            db.create_collection('time_series_models')
        
        model_doc = {
            'model_name': f'TimeSeries_{model_type}',
            'model_type': 'TimeSeries',
            'model_version': datetime.now().strftime('%Y%m%d_%H%M'),
            'feature_version': feature_version,
            'model_path': f'time_series/{model_type}_{datetime.now().strftime("%Y%m%d_%H%M")}.pkl',
            'metrics': metrics,
            'parameters': {'alpha': 0.3} if model_type == 'exponential_smoothing' else {},
            'created_at': datetime.now(),
            'status': 'trained',
            'city': 'Karachi',
            'prediction_horizon': '72h',
            'forecasts_generated': forecasts_count
        }
        
        # Save model file locally
        os.makedirs('models/time_series', exist_ok=True)
        model_path = f'models/time_series/{model_type}_{datetime.now().strftime("%Y%m%d_%H%M")}.pkl'
        
        # Create simple model data
        model_data = {
            'model_type': model_type,
            'created_at': datetime.now(),
            'parameters': model_doc['parameters'],
            'metrics': metrics
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        model_doc['model_path'] = model_path
        
        # Save to registry
        result = db['time_series_models'].insert_one(model_doc)
        
        print(f"✅ Time series model saved to Model Registry: {result.inserted_id}")
        
        client.close()
        return result.inserted_id
        
    except Exception as e:
        print(f"⚠️ Could not save to Model Registry: {str(e)[:100]}")
        return None

def main():
    """Main without Prophet"""
    print("=" * 60)
    print("📈 FIXED TIME-SERIES FORECAST (No Prophet)")
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
    ma_24h = df['aqi'].tail(24).mean() if len(df) >= 24 else df['aqi'].mean()
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
            'model': 'moving_average',
            'hour': hour
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
        record['model'] = 'timeseries_ensemble'
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE', 'aqi_predictor')]
    
    # Save to various collections for dashboard compatibility
    db.timeseries_forecasts_3day.delete_many({})
    if ensemble_records:
        db.timeseries_forecasts_3day.insert_many(ensemble_records)
    
    # Also save to ensemble_predictions for backward compatibility
    db.ensemble_predictions.delete_many({})
    if ensemble_records:
        db.ensemble_predictions.insert_many(ensemble_records)
    
    client.close()
    
    print(f"\n💾 Saved ensemble forecast for {len(ensemble_records)} days")
    
    # Save to Model Registry
    metrics = {
        'forecasts_generated': len(all_preds),
        'models_used': ['seasonal', 'exponential_smoothing', 'moving_average'],
        'avg_aqi': daily_avg['predicted_aqi'].mean(),
        'date_range': f"{daily_avg['date'].min()} to {daily_avg['date'].max()}"
    }
    
    save_time_series_to_registry('ensemble', metrics, len(all_preds))
    
    print(f"\n✅ Time series forecasting complete!")
    print(f"📊 3-day forecasts saved to database")
    print(f"📋 Model saved to Model Registry")

if __name__ == "__main__":
    main()