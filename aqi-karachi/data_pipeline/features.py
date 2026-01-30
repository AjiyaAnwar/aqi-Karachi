"""
features_simple.py - Simple CORRECT feature engineering
"""
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

def create_simple_features():
    print("=" * 60)
    print("🌫️ SIMPLE Feature Engineering (3h & 6h targets)")
    print("=" * 60)
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    main_db = client['aqi_predictor']
    
    # Load raw data
    cursor = list(main_db.aqi_measurements.find(
        {}, 
        {'_id': 0, 'timestamp': 1, 'aqi': 1}
    ).sort('timestamp', 1))
    
    df = pd.DataFrame(cursor)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    
    # Resample to hourly
    df_hourly = df['aqi'].resample('h').mean()
    df_hourly = df_hourly.interpolate()
    
    # Create simple features
    features = pd.DataFrame(index=df_hourly.index)
    features['aqi'] = df_hourly
    
    # Time features
    features['hour'] = features.index.hour
    features['day_of_week'] = features.index.dayofweek
    
    # Simple lag features
    features['lag_1h'] = df_hourly.shift(1)
    features['lag_3h'] = df_hourly.shift(3)
    features['lag_6h'] = df_hourly.shift(6)
    features['lag_24h'] = df_hourly.shift(24)
    
    # TARGETS - Use 3h and 6h (easier, better correlation)
    features['target_3h'] = df_hourly.shift(-3)
    features['target_6h'] = df_hourly.shift(-6)
    
    # Clean
    features = features.dropna()
    
    # Check correlations
    print("\\n🎯 TARGET CORRELATIONS:")
    print(f"target_3h correlation: {features['aqi'].corr(features['target_3h']):.4f}")
    print(f"target_6h correlation: {features['aqi'].corr(features['target_6h']):.4f}")
    
    # Save to Feature Store
    fs_db = client['aqi_feature_store']
    
    # Clear old
    if 'aqi_features_simple' in fs_db.list_collection_names():
        fs_db['aqi_features_simple'].delete_many({})
    
    # Prepare records
    records = features.reset_index().to_dict('records')
    for record in records:
        record['timestamp'] = record['timestamp'].isoformat()
        record['created_at'] = datetime.now().isoformat()
    
    if records:
        fs_db['aqi_features_simple'].insert_many(records)
        print(f"\\n💾 Saved {len(records)} records to 'aqi_features_simple'")
    
    client.close()
    
    print("\\n✅ Simple features created with CORRECT targets!")
    return features

if __name__ == "__main__":
    create_simple_features()