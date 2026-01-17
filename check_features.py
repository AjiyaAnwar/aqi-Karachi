"""
Verify features were created correctly
"""
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np  # Added missing import

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('MONGODB_DATABASE')]

# Check counts
raw_count = db.aqi_measurements.count_documents({})
features_count = db.aqi_features.count_documents({})

print("📊 DATABASE STATUS:")
print(f"Raw measurements: {raw_count}")
print(f"Feature records:  {features_count}")

if features_count > 0:
    # Get one sample
    sample = db.aqi_features.find_one()
    print(f"\n📝 Sample document has {len(sample)} fields")
    
    # Get feature names (excluding metadata)
    exclude_fields = ['_id', 'timestamp', 'created_at', 'feature_version', 'total_features']
    feature_names = [k for k in sample.keys() if k not in exclude_fields]
    
    print(f"\n🔧 {len(feature_names)} ML Features:")
    for i, feat in enumerate(sorted(feature_names), 1):
        print(f"  {i:2d}. {feat}")
    
    # Load into DataFrame for analysis
    all_features = list(db.aqi_features.find({}, {'_id': 0}))
    df = pd.DataFrame(all_features)
    
    if not df.empty and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        print(f"\n📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"📈 Total records: {len(df)}")
        
        # Days of data
        days = (df['timestamp'].max() - df['timestamp'].min()).days
        print(f"📆 Days of data: {days} days")
        
        # Check for missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missing = df[numeric_cols].isna().sum()
        
        print(f"\n✅ Data Quality Check:")
        print(f"  Complete records: {(~df[numeric_cols].isna().any(axis=1)).sum()}/{len(df)}")
        
        if missing.sum() > 0:
            print(f"⚠️  Missing values found:")
            for col in missing[missing > 0].index:
                print(f"    {col}: {missing[col]} missing")
        else:
            print(f"  All numeric features complete ✓")
    
    # Check target variable
    if 'target_24h' in df.columns:
        print(f"\n🎯 Target variable 'target_24h' (AQI 24h ahead):")
        print(f"  Mean: {df['target_24h'].mean():.1f}")
        print(f"  Std:  {df['target_24h'].std():.1f}")
        print(f"  Min:  {df['target_24h'].min():.1f}")
        print(f"  Max:  {df['target_24h'].max():.1f}")
        
        # Check AQI distribution
        print(f"\n📊 AQI Distribution:")
        print(f"  Current AQI mean: {df['aqi'].mean():.1f}")
        print(f"  PM2.5 mean: {df['pm25'].mean():.1f} µg/m³")
        print(f"  PM10 mean: {df['pm10'].mean():.1f} µg/m³")
    
    # Show correlation with target
    if 'target_24h' in df.columns and 'aqi' in df.columns:
        correlation = df['aqi'].corr(df['target_24h'])
        print(f"\n📈 Correlation current AQI vs next 24h AQI: {correlation:.3f}")
        
        # Check lag correlations
        for lag in ['lag_1', 'lag_2', 'lag_3', 'lag_24']:
            if lag in df.columns:
                corr = df[lag].corr(df['target_24h'])
                print(f"  {lag}: {corr:.3f}")

else:
    print("❌ No features found in database")

client.close()

print(f"\n💾 CSV files available:")
print(f"  data/features.csv - All features")
print(f"  data/features_sample.csv - Sample")
print(f"  data/features_final.csv - Simple features")
