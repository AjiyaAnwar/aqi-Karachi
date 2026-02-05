"""
features_simple.py - Simple CORRECT feature engineering with proper imports
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from data_pipeline
sys.path.append(project_root)

load_dotenv()

def create_simple_features():
    print("=" * 60)
    print("🌫️ SIMPLE Feature Engineering (3h & 6h targets)")
    print("=" * 60)
    
    try:
        # Try to import MongoDB Manager
        from cicd.mongodb_utils import MongoDBManager
        use_mongo_manager = True
        print("✅ Using MongoDB Manager")
    except ImportError as e:
        print(f"⚠️ MongoDB Manager not found: {e}")
        print("⚠️ Falling back to direct MongoDB connection")
        use_mongo_manager = False
        from pymongo import MongoClient
    
    try:
        if use_mongo_manager:
            mongo_manager = MongoDBManager()
            mongo_manager.log_pipeline_step("feature_engineering", "started")
            
            # Get raw data using manager
            features_df = mongo_manager.get_latest_raw_data(limit_hours=168)  # Last 7 days
            
            if features_df.empty:
                print("⚠️ No data from MongoDB Manager, using direct connection")
                use_mongo_manager = False
        else:
            features_df = pd.DataFrame()
        
        if not use_mongo_manager:
            # Fallback to direct connection
            mongodb_uri = os.getenv('MONGODB_URI')
            client = MongoClient(mongodb_uri)
            main_db = client['aqi_predictor']
            
            cursor = list(main_db.aqi_measurements.find(
                {}, 
                {'_id': 0, 'timestamp': 1, 'aqi': 1, 'pm25': 1, 'pm10': 1}
            ).sort('timestamp', 1))
            
            features_df = pd.DataFrame(cursor)
            client.close()
        
        if features_df.empty:
            print("❌ No data available for feature engineering")
            return None
        
        # Process data
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
        features_df = features_df.sort_values('timestamp')
        features_df.set_index('timestamp', inplace=True)
        
        # Resample to hourly
        df_hourly = features_df['aqi'].resample('h').mean()
        df_hourly = df_hourly.interpolate()
        
        # Create simple features
        features = pd.DataFrame(index=df_hourly.index)
        features['aqi'] = df_hourly
        
        # Add other pollutants if available
        if 'pm25' in features_df.columns:
            features['pm25'] = features_df['pm25'].resample('h').mean().interpolate()
        if 'pm10' in features_df.columns:
            features['pm10'] = features_df['pm10'].resample('h').mean().interpolate()
        
        # Time features
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['month'] = features.index.month
        features['is_peak_hour'] = features.index.hour.isin([8, 9, 17, 18]).astype(int)
        
        # Lag features
        features['lag_1h'] = df_hourly.shift(1)
        features['lag_3h'] = df_hourly.shift(3)
        features['lag_6h'] = df_hourly.shift(6)
        features['lag_24h'] = df_hourly.shift(24)
        
        # Rolling statistics
        features['rolling_avg_3h'] = df_hourly.rolling(window=3).mean()
        features['rolling_avg_6h'] = df_hourly.rolling(window=6).mean()
        features['rolling_std_3h'] = df_hourly.rolling(window=3).std()
        
        # TARGETS - Use 3h and 6h (easier, better correlation)
        features['target_3h'] = df_hourly.shift(-3)
        features['target_6h'] = df_hourly.shift(-6)
        features['target_24h'] = df_hourly.shift(-24)  # For daily forecasting
        
        # Clean
        features = features.dropna()
        
        # Check correlations
        print("\n🎯 TARGET CORRELATIONS:")
        if 'target_3h' in features.columns:
            print(f"target_3h correlation: {features['aqi'].corr(features['target_3h']):.4f}")
        if 'target_6h' in features.columns:
            print(f"target_6h correlation: {features['aqi'].corr(features['target_6h']):.4f}")
        if 'target_24h' in features.columns:
            print(f"target_24h correlation: {features['aqi'].corr(features['target_24h']):.4f}")
        
        print(f"\n📊 Features shape: {features.shape}")
        print(f"📅 Date range: {features.index.min()} to {features.index.max()}")
        
        # Save features
        if use_mongo_manager:
            # Save to Feature Store with versioning
            version, feature_id = mongo_manager.store_features_with_versioning(
                features_df=features.reset_index(),
                description="Karachi AQI features with 3h/6h/24h targets",
                is_training=True
            )
            
            print(f"💾 Saved features version: {version}")
            mongo_manager.log_pipeline_step("feature_engineering", "completed", {
                "feature_count": len(features.columns),
                "record_count": len(features),
                "version": version
            })
        else:
            # Fallback: Save directly to MongoDB
            mongodb_uri = os.getenv('MONGODB_URI')
            client = MongoClient(mongodb_uri)
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
                print(f"💾 Saved {len(records)} records to 'aqi_features_simple'")
            
            client.close()
        
        print(f"📊 Total records: {len(features)}")
        
        return features
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    features = create_simple_features()
    if features is not None:
        print("\n✅ Simple features created with CORRECT targets!")
    else:
        print("\n❌ Feature engineering failed!")
