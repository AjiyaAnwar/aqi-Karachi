"""
DEBUG DATA LEAKAGE - Find why R² is negative
"""
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def debug_features():
    """Debug feature engineering issues"""
    print("=" * 70)
    print("🔍 DEBUGGING FEATURE LEAKAGE")
    print("=" * 70)
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    
    # 1. Check Feature Store
    print("\n1️⃣ CHECKING FEATURE STORE")
    print("-" * 40)
    
    fs_db = client['aqi_feature_store']
    
    if 'aqi_features' not in fs_db.list_collection_names():
        print("❌ No features in feature store")
        return
    
    # Get features
    cursor = list(fs_db['aqi_features'].find({}, {'_id': 0}).limit(100))
    df_features = pd.DataFrame(cursor)
    
    print(f"📊 Features loaded: {len(df_features)} rows, {len(df_features.columns)} columns")
    
    # Check target
    if 'target_24h' in df_features.columns:
        print(f"\n🎯 TARGET ANALYSIS (target_24h):")
        print(f"   Mean: {df_features['target_24h'].mean():.2f}")
        print(f"   Std: {df_features['target_24h'].std():.2f}")
        print(f"   Min: {df_features['target_24h'].min():.2f}")
        print(f"   Max: {df_features['target_24h'].max():.2f}")
        print(f"   NaN: {df_features['target_24h'].isna().sum()}")
        
        # Check correlation with AQI
        if 'aqi' in df_features.columns:
            correlation = df_features['aqi'].corr(df_features['target_24h'])
            print(f"   Correlation with AQI: {correlation:.4f}")
            
            # This is CRITICAL: target_24h should be aqi shifted by -24
            # So aqi at time t should correlate with target_24h at time t-24
            # Let's check if there's data leakage
            print(f"\n🔍 CHECKING FOR DATA LEAKAGE:")
            
            # Shift aqi to compare
            df_features['aqi_shifted'] = df_features['aqi'].shift(24)
            valid_mask = df_features[['aqi_shifted', 'target_24h']].notna().all(axis=1)
            
            if valid_mask.any():
                leakage_corr = df_features.loc[valid_mask, 'aqi_shifted'].corr(
                    df_features.loc[valid_mask, 'target_24h']
                )
                print(f"   Correlation (aqi_shifted_24h vs target_24h): {leakage_corr:.4f}")
                
                if leakage_corr > 0.9:
                    print("   ⚠️  HIGH CORRELATION - Possible data leakage!")
                    print("   💡 Target might be using current/future data")
                else:
                    print("   ✅ Low correlation - no obvious leakage")
            else:
                print("   ❌ Cannot check leakage - not enough data")
    
    # 2. Check Raw Data
    print("\n2️⃣ CHECKING RAW DATA")
    print("-" * 40)
    
    main_db = client['aqi_predictor']
    cursor = list(main_db['aqi_measurements'].find(
        {}, 
        {'_id': 0, 'timestamp': 1, 'aqi': 1}
    ).sort('timestamp', 1))
    
    if cursor:
        df_raw = pd.DataFrame(cursor)
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
        df_raw = df_raw.sort_values('timestamp')
        
        print(f"📊 Raw data: {len(df_raw)} rows")
        print(f"📅 Time range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
        
        # Check if we can create proper target
        df_raw['aqi_next_24h'] = df_raw['aqi'].shift(-24)
        df_raw = df_raw.dropna(subset=['aqi_next_24h'])
        
        print(f"✅ Valid rows with 24h future: {len(df_raw)}")
        
        # Check correlation
        correlation_raw = df_raw['aqi'].corr(df_raw['aqi_next_24h'])
        print(f"🎯 Correlation (current vs next 24h): {correlation_raw:.4f}")
        
        if correlation_raw < 0.3:
            print("   ⚠️  Low correlation - AQI changes a lot in 24h")
            print("   💡 Try predicting 3h or 6h instead of 24h")
    
    # 3. Check Model Registry
    print("\n3️⃣ CHECKING MODEL REGISTRY")
    print("-" * 40)
    
    mr_db = client['aqi_model_registry']
    
    if 'models' in mr_db.list_collection_names():
        latest_model = mr_db['models'].find_one(sort=[('created_at', -1)])
        if latest_model:
            print(f"📋 Latest model: {latest_model.get('model_name', 'Unknown')}")
            print(f"📊 Metrics: {latest_model.get('metrics', {})}")
    
    client.close()
    
    print("\n" + "=" * 70)
    print("🔍 DEBUG SUMMARY")
    print("=" * 70)
    
    print("\n🎯 MOST LIKELY ISSUES:")
    print("1. Your features.py creates target_24h but train_models.py uses 'target'")
    print("2. train_models.py creates its own features differently")
    print("3. 24h prediction might be too hard - try 3h or 6h")
    
    print("\n💡 RECOMMENDATIONS:")
    print("1. Make train_models.py use features from Feature Store (not create its own)")
    print("2. Try shorter prediction horizon (3h or 6h)")
    print("3. Check if target_24h in Feature Store is correct")

if __name__ == "__main__":
    debug_features()