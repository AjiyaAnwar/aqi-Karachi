"""
VERIFY Feature Store Fix
"""
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def verify_feature_store():
    print("=" * 70)
    print("🔍 VERIFYING FEATURE STORE FIX")
    print("=" * 70)
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    
    # Check Feature Store
    print("\n1️⃣ CHECKING FEATURE STORE")
    print("-" * 40)
    
    fs_db = client['aqi_feature_store']
    
    if 'aqi_features' not in fs_db.list_collection_names():
        print("❌ No features in Feature Store")
        return
    
    # Get a sample
    sample = fs_db['aqi_features'].find_one({}, {'_id': 0})
    if sample:
        print(f"📊 Sample feature record:")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Has 'target_24h'? {'target_24h' in sample}")
        print(f"  Has 'pm25'? {'pm25' in sample}")
        print(f"  Has 'lag_' features? {any('lag_' in key for key in sample.keys())}")
        
        # Count features vs raw data
        all_features = list(fs_db['aqi_features'].find({}, {'_id': 0}).limit(10))
        if all_features:
            df = pd.DataFrame(all_features)
            print(f"\n📈 Feature Store Statistics:")
            print(f"  Total columns: {len(df.columns)}")
            print(f"  Target columns: {[c for c in df.columns if 'target' in c]}")
            print(f"  Lag columns: {[c for c in df.columns if 'lag_' in c]}")
            print(f"  Rolling columns: {[c for c in df.columns if 'rolling_' in c]}")
            
            # Check if it looks like features or raw data
            if 'target_24h' in df.columns and any('lag_' in c for c in df.columns):
                print("✅ Feature Store looks GOOD - contains engineered features")
            else:
                print("❌ Feature Store looks BAD - contains raw data, not features!")
    
    # Check Raw Data
    print("\n2️⃣ CHECKING RAW DATA")
    print("-" * 40)
    
    main_db = client['aqi_predictor']
    if 'aqi_measurements' in main_db.list_collection_names():
        count = main_db['aqi_measurements'].count_documents({})
        print(f"  Raw data records: {count:,}")
    
    if 'aqi_features' in main_db.list_collection_names():
        count = main_db['aqi_features'].count_documents({})
        print(f"  ⚠️  WARNING: aqi_features collection in main DB: {count:,} records")
        print("  💡 This should be empty or moved to feature store!")
    
    client.close()

if __name__ == "__main__":
    verify_feature_store()