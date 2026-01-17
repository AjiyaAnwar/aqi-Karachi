"""
Simplified feature engineering - step by step
"""
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def create_features_simple():
    """Simplified feature engineering pipeline"""
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE')]
    
    print("Step 1: Fetching data...")
    # Get data as list, exclude _id
    cursor = list(db.aqi_measurements.find(
        {}, 
        {'_id': 0, 'timestamp': 1, 'aqi': 1, 'pm2_5': 1, 'pm10': 1}
    ).sort("timestamp", 1))
    
    if not cursor:
        print("No data found")
        return
    
    print(f"Found {len(cursor)} records")
    
    print("\nStep 2: Creating DataFrame...")
    df = pd.DataFrame(cursor)
    
    print("\nStep 3: Converting timestamps...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    print(f"Valid records: {len(df)}")
    
    print("\nStep 4: Sorting and setting index...")
    df = df.sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    
    print("\nStep 5: Converting numeric columns...")
    for col in ['aqi', 'pm2_5', 'pm10']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("\nStep 6: Resampling to hourly...")
    # Group by hour and take mean
    df_hourly = df.resample('h').mean()
    df_hourly = df_hourly.interpolate(method='time')
    
    print(f"Hourly records: {len(df_hourly)}")
    
    print("\nStep 7: Creating basic features...")
    features = pd.DataFrame(index=df_hourly.index)
    
    # Basic features
    features['aqi'] = df_hourly['aqi']
    features['pm25'] = df_hourly['pm2_5']
    features['pm10'] = df_hourly['pm10']
    
    # Time features
    features['hour'] = df_hourly.index.hour
    features['day_of_week'] = df_hourly.index.dayofweek
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
    
    # Simple lags
    for lag in [1, 2, 3, 24]:
        features[f'lag_{lag}'] = df_hourly['aqi'].shift(lag)
    
    # Simple rolling stats
    features['rolling_24h_mean'] = df_hourly['aqi'].rolling(24, min_periods=1).mean()
    
    # Target (next 24h)
    features['target_24h'] = df_hourly['aqi'].shift(-24)
    
    # Drop rows with missing values
    features = features.dropna()
    
    print(f"\n✅ Final feature shape: {features.shape}")
    print(f"Features: {list(features.columns)}")
    
    print("\nStep 8: Saving to MongoDB...")
    features_collection = db['aqi_features']
    features_collection.delete_many({})
    
    # Prepare records
    features_reset = features.reset_index()
    features_reset['timestamp'] = features_reset['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    records = features_reset.to_dict('records')
    for record in records:
        record['created_at'] = datetime.now().isoformat()
    
    result = features_collection.insert_many(records)
    print(f"Saved {len(result.inserted_ids)} feature records")
    
    # Save sample to CSV
    features.head(100).to_csv('data/features_sample.csv')
    print("Sample saved to data/features_sample.csv")
    
    client.close()
    print("\n🎉 Done!")
    
    return features

if __name__ == "__main__":
    create_features_simple()
