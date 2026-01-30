"""
DEBUG: Why is target_24h correlation negative?
"""
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

def debug_target_creation():
    print("=" * 70)
    print("🔍 DEBUG: Why is target_24h correlation negative?")
    print("=" * 70)
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    main_db = client['aqi_predictor']
    
    # Get raw data
    cursor = list(main_db.aqi_measurements.find(
        {}, 
        {'_id': 0, 'timestamp': 1, 'aqi': 1}
    ).sort('timestamp', 1))
    
    df = pd.DataFrame(cursor)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    
    print(f"📊 Raw data: {len(df)} records")
    print(f"📅 Time range: {df.index.min()} to {df.index.max()}")
    
    # Step 1: Check raw data correlation
    print("\n1️⃣ RAW DATA CORRELATION")
    print("-" * 40)
    
    # What should target_24h be?
    df['aqi_24h_later'] = df['aqi'].shift(-24)
    df_valid = df.dropna(subset=['aqi_24h_later'])
    
    raw_corr = df_valid['aqi'].corr(df_valid['aqi_24h_later'])
    print(f"Raw data correlation (current vs 24h later): {raw_corr:.4f}")
    
    # Show examples
    print("\n📋 Example pairs (current vs 24h later):")
    for i in range(5):
        current_time = df_valid.index[i]
        current_aqi = df_valid['aqi'].iloc[i]
        future_aqi = df_valid['aqi_24h_later'].iloc[i]
        future_time = current_time + timedelta(hours=24)
        
        print(f"  {current_time}: {current_aqi:.1f} → {future_time}: {future_aqi:.1f}")
    
    # Step 2: Check what features.py is doing
    print("\n2️⃣ CHECKING FEATURES.PY OUTPUT")
    print("-" * 40)
    
    fs_db = client['aqi_feature_store']
    if 'aqi_features' in fs_db.list_collection_names():
        cursor = list(fs_db['aqi_features'].find(
            {}, 
            {'_id': 0, 'timestamp': 1, 'aqi': 1, 'target_24h': 1}
        ).limit(5))
        
        df_features = pd.DataFrame(cursor)
        df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
        
        print("Feature Store samples:")
        for i, row in df_features.iterrows():
            print(f"  {row['timestamp']}: AQI={row['aqi']}, target_24h={row['target_24h']}")
        
        # Compare with raw data
        print("\n🔍 COMPARISON with raw data:")
        for i, row in df_features.iterrows():
            timestamp = row['timestamp']
            aqi_in_features = row['aqi']
            target_in_features = row['target_24h']
            
            # What should target_24h actually be?
            expected_target_time = timestamp + timedelta(hours=24)
            
            # Find AQI at expected time in raw data
            expected_target_aqi = None
            if expected_target_time in df.index:
                expected_target_aqi = df.loc[expected_target_time, 'aqi']
            
            print(f"  {timestamp}:")
            print(f"    AQI in features: {aqi_in_features}")
            print(f"    target_24h in features: {target_in_features}")
            print(f"    Expected (AQI at {expected_target_time}): {expected_target_aqi}")
            print(f"    Match? {'✅' if expected_target_aqi == target_in_features else '❌'}")
    
    # Step 3: The REAL problem - maybe data is REVERSED?
    print("\n3️⃣ CHECKING FOR REVERSED DATA")
    print("-" * 40)
    
    # Check if data is in correct chronological order
    print("Checking chronological order...")
    
    # Get first and last few timestamps
    first_5 = df.index[:5]
    last_5 = df.index[-5:]
    
    print(f"First 5 timestamps: {[t.strftime('%Y-%m-%d %H:%M') for t in first_5]}")
    print(f"Last 5 timestamps: {[t.strftime('%Y-%m-%d %H:%M') for t in last_5]}")
    
    # Check if they're in correct order
    is_ascending = all(first_5[i] < first_5[i+1] for i in range(len(first_5)-1))
    print(f"Timestamps ascending? {is_ascending}")
    
    # Check AQI pattern
    print(f"\nAQI pattern in first 24 hours:")
    first_24h = df.head(24)
    for i in range(0, 24, 6):
        time = first_24h.index[i]
        aqi = first_24h['aqi'].iloc[i]
        print(f"  {time.strftime('%H:%M')}: {aqi:.1f}")
    
    client.close()
    
    print("\n" + "=" * 70)
    print("🎯 MOST LIKELY ISSUE:")
    print("=" * 70)
    print("Your AQI data from OpenMeteo might be:")
    print("1. In REVERSE chronological order")
    print("2. Has timestamp issues (timezone wrong)")
    print("3. Or features.py shift(-24) is wrong")
    
    print("\n💡 QUICK FIX: Try 3h prediction instead")
    print("   Shorter horizon = easier prediction = better R²")

def fix_features_for_3h():
    """Show how to modify features.py for 3h prediction"""
    print("\n" + "=" * 70)
    print("🔧 QUICK FIX: Modify features.py for 3h prediction")
    print("=" * 70)
    
    fix_code = '''
# IN features.py, change these lines:

# OLD (line ~180-185):
target_24h = df_hourly_complete.shift(-24)
target_72h = df_hourly_complete.shift(-72)
features['target_24h'] = target_24h_aligned
features['target_72h'] = target_72h_aligned

# NEW (predict 3h and 6h instead):
target_3h = df_hourly_complete.shift(-3)
target_6h = df_hourly_complete.shift(-6)
features['target_3h'] = target_3h.reindex(df_hourly.index)
features['target_6h'] = target_6h.reindex(df_hourly.index)

# Also update train_models.py to use target_3h instead of target_24h
    '''
    
    print(fix_code)

if __name__ == "__main__":
    debug_target_creation()
    fix_features_for_3h()