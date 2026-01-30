"""
Simplified feature engineering - step by step
UPDATED: Saves to feature store database
"""
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def create_features_simple():
    """Simplified feature engineering pipeline - UPDATED FOR FEATURE STORE"""
    client = MongoClient(os.getenv('MONGODB_URI'))
    
    # Connect to feature store database
    feature_store_db = os.getenv('FEATURE_STORE_DB', 'aqi_feature_store')
    db = client[feature_store_db]
    
    print(f"📊 Connecting to Feature Store: {feature_store_db}")
    
    # Get raw data from main database
    main_db = client[os.getenv('MONGODB_DATABASE', 'aqi_predictor')]
    
    print("Step 1: Fetching raw data from main database...")
    
    # First, check what columns exist in the collection
    sample_record = main_db.aqi_measurements.find_one()
    if sample_record:
        print(f"Available columns: {list(sample_record.keys())}")
    
    # Try different column name variations
    projection = {'_id': 0, 'timestamp': 1, 'aqi': 1}
    
    # Common variations of pm2.5 column names
    pm25_variations = ['pm2_5', 'pm25', 'pm2.5', 'pm2_5_concentration', 'pm25_concentration']
    pm10_variations = ['pm10', 'pm10_concentration']
    
    # Check which columns actually exist
    cursor = list(main_db.aqi_measurements.find({}).limit(1))
    if cursor:
        record = cursor[0]
        # Add existing columns to projection
        for pm25_var in pm25_variations:
            if pm25_var in record:
                projection[pm25_var] = 1
                print(f"Found PM2.5 column: {pm25_var}")
                break
        
        for pm10_var in pm10_variations:
            if pm10_var in record:
                projection[pm10_var] = 1
                print(f"Found PM10 column: {pm10_var}")
                break
    
    # Get all data
    cursor = list(main_db.aqi_measurements.find({}, projection).sort("timestamp", 1))
    
    if not cursor:
        print("No data found")
        return
    
    print(f"Found {len(cursor)} records")
    
    print("\nStep 2: Creating DataFrame...")
    df = pd.DataFrame(cursor)
    
    print(f"Columns in DataFrame: {list(df.columns)}")
    
    print("\nStep 3: Converting timestamps...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    print(f"Valid records: {len(df)}")
    
    print("\nStep 4: Sorting and setting index...")
    df = df.sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    
    print("\nStep 5: Converting numeric columns...")
    # Convert all numeric columns except timestamp
    for col in df.columns:
        if col not in ['timestamp']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"  Converted {col}")
            except:
                print(f"  Skipped {col} (not numeric)")
    
    # Standardize column names
    print("\nStep 5.5: Standardizing column names...")
    column_mapping = {}
    
    # Find and rename PM2.5 column
    for old_name in ['pm2_5', 'pm25', 'pm2.5', 'pm2_5_concentration', 'pm25_concentration']:
        if old_name in df.columns:
            column_mapping[old_name] = 'pm25'
            print(f"  Renaming {old_name} -> pm25")
            break
    
    # Find and rename PM10 column
    for old_name in ['pm10', 'pm10_concentration']:
        if old_name in df.columns:
            column_mapping[old_name] = 'pm10'
            print(f"  Renaming {old_name} -> pm10")
            break
    
    df = df.rename(columns=column_mapping)
    
    print(f"Final columns: {list(df.columns)}")
    
    print("\nStep 6: Resampling to hourly...")
    # Group by hour and take mean
    df_hourly = df.resample('h').mean()
    
    # Count non-nan values before interpolation
    print(f"Records before interpolation: {len(df_hourly)}")
    print(f"Missing values before interpolation:")
    for col in df_hourly.columns:
        missing = df_hourly[col].isna().sum()
        if missing > 0:
            print(f"  {col}: {missing} missing ({missing/len(df_hourly)*100:.1f}%)")
    
    # Interpolate missing values
    df_hourly = df_hourly.interpolate(method='time', limit_direction='both')
    
    print(f"Hourly records: {len(df_hourly)}")
    
    print("\nStep 7: Creating enhanced features...")
    features = pd.DataFrame(index=df_hourly.index)
    
    # Basic features
    features['aqi'] = df_hourly['aqi']
    
    # Add PM2.5 if available
    if 'pm25' in df_hourly.columns:
        features['pm25'] = df_hourly['pm25']
        print("  Added pm25 feature")
    else:
        # Create placeholder if missing
        features['pm25'] = df_hourly['aqi'] * 0.5  # Rough estimate
        print("  WARNING: pm25 not found, using estimate")
    
    # Add PM10 if available
    if 'pm10' in df_hourly.columns:
        features['pm10'] = df_hourly['pm10']
        print("  Added pm10 feature")
    else:
        # Create placeholder if missing
        features['pm10'] = df_hourly['aqi'] * 0.8  # Rough estimate
        print("  WARNING: pm10 not found, using estimate")
    
    # Time features
    features['hour'] = df_hourly.index.hour
    features['day_of_week'] = df_hourly.index.dayofweek
    features['month'] = df_hourly.index.month
    
    print(f"  Created time features: hour, day_of_week, month")
    
    # Cyclical encoding
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
    features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    
    print(f"  Created cyclical features: hour_sin, hour_cos, day_sin, day_cos")
    
    # Interaction features (with safeguards)
    features['aqi_pm25_ratio'] = features['aqi'] / (features['pm25'] + 1)
    features['aqi_pm10_ratio'] = features['aqi'] / (features['pm10'] + 1)
    
    print(f"  Created interaction features")
    
    # Lag features with 'h' suffix
    if 'aqi' in df_hourly.columns:
        for lag in [1, 3, 6, 12, 24]:
            features[f'lag_{lag}h'] = df_hourly['aqi'].shift(lag)
            print(f"  Created lag_{lag}h feature")
    else:
        print("  WARNING: AQI column missing for lag features")
    
    # Rolling statistics
    if 'aqi' in df_hourly.columns:
        for window in [6, 12, 24, 72]:
            features[f'rolling_mean_{window}h'] = df_hourly['aqi'].rolling(window=window, min_periods=1).mean()
            features[f'rolling_std_{window}h'] = df_hourly['aqi'].rolling(window=window, min_periods=1).std()
            features[f'rolling_max_{window}h'] = df_hourly['aqi'].rolling(window=window, min_periods=1).max()
            print(f"  Created rolling features for {window}h window")
    
    # Target (next 24h) - MUST BE NAMED target_24h
    if 'aqi' in df_hourly.columns:
        features['target_24h'] = df_hourly['aqi'].shift(-24)
        print("  Created target_24h feature")
    else:
        print("  ERROR: Cannot create target without AQI data")
        return
    
    print("\nStep 8: Cleaning data...")
    
    # Fill lag features
    lag_cols = [col for col in features.columns if 'lag_' in col]
    for col in lag_cols:
        features[col] = features[col].fillna(method='bfill').fillna(method='ffill')
    print(f"  Filled {len(lag_cols)} lag features")
    
    # Fill rolling features
    rolling_cols = [col for col in features.columns if 'rolling_' in col]
    for col in rolling_cols:
        features[col] = features[col].fillna(method='bfill').fillna(method='ffill')
    print(f"  Filled {len(rolling_cols)} rolling features")
    
    # Drop rows with missing target (last 24 hours)
    initial_count = len(features)
    features = features.dropna(subset=['target_24h'])
    dropped_count = initial_count - len(features)
    print(f"  Dropped {dropped_count} rows with missing target")
    
    # Fill any remaining NaN with column mean
    nan_before = features.isna().sum().sum()
    features = features.fillna(features.mean())
    nan_after = features.isna().sum().sum()
    print(f"  Filled {nan_before - nan_after} remaining NaN values")
    
    print(f"\n✅ Final feature shape: {features.shape}")
    print(f"Features: {list(features.columns)}")
    print(f"Samples after cleaning: {len(features)}")
    
    # Show data summary
    print("\n📊 Data Summary:")
    print(f"Time range: {features.index.min()} to {features.index.max()}")
    print(f"Total hours: {(features.index.max() - features.index.min()).total_seconds() / 3600:.0f}")
    print(f"Average AQI: {features['aqi'].mean():.1f}")
    print(f"Target range: {features['target_24h'].min():.1f} to {features['target_24h'].max():.1f}")
    
    print("\nStep 9: Saving to Feature Store...")
    features_collection = db['aqi_features']
    
    # Clear existing features
    features_collection.delete_many({})
    print("  Cleared existing features")
    
    # Prepare records - limit to 5000 for performance
    features_reset = features.reset_index()
    
    # Convert datetime to string
    features_reset['timestamp'] = features_reset['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Limit number of records to insert (MongoDB has limits)
    if len(features_reset) > 5000:
        print(f"  Warning: {len(features_reset)} records found, limiting to 5000")
        features_reset = features_reset.head(5000)
    
    records = features_reset.to_dict('records')
    
    # Add metadata
    for record in records:
        record['created_at'] = datetime.now().isoformat()
        record['feature_version'] = '2.0'
        record['source_database'] = os.getenv('MONGODB_DATABASE', 'aqi_predictor')
        record['feature_store'] = feature_store_db
    
    # Insert in batches
    batch_size = 1000
    total_inserted = 0
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        result = features_collection.insert_many(batch)
        total_inserted += len(result.inserted_ids)
        print(f"  Inserted batch {i//batch_size + 1}: {len(batch)} records")
    
    print(f"💾 Saved {total_inserted} feature records to Feature Store")
    
    # Also create a feature version entry
    version_collection = db['feature_versions']
    version = datetime.now().strftime("%Y%m%d_%H%M")
    
    version_doc = {
        'version': version,
        'timestamp': datetime.now(),
        'feature_count': len(features.columns),
        'record_count': total_inserted,
        'feature_names': list(features.columns),
        'time_range': {
            'start': features.index.min().isoformat(),
            'end': features.index.max().isoformat()
        }
    }
    
    version_collection.insert_one(version_doc)
    print(f"🏷️  Created feature version: {version}")
    
    # Save sample to CSV
    os.makedirs('data', exist_ok=True)
    features.head(100).to_csv('data/features_sample.csv', index=False)
    print("📁 Sample saved to data/features_sample.csv")
    
    # Save full dataset for training
    features.to_csv('data/features_full.csv', index=True)
    print("📁 Full dataset saved to data/features_full.csv")
    
    client.close()
    print("\n🎉 Feature engineering complete! Features saved to Feature Store.")
    
    return features

if __name__ == "__main__":
    print("=" * 60)
    print("🌫️ AQI Karachi - Feature Engineering (Feature Store)")
    print("=" * 60)
    create_features_simple()