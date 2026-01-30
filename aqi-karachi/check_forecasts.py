#!/usr/bin/env python
"""
Debug script to check where forecasts are being saved
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def main():
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        print("❌ MONGODB_URI not set")
        return
    
    client = MongoClient(mongodb_uri)
    
    databases = [
        os.getenv("MONGODB_DATABASE", "aqi_predictor"),
        "aqi_model_registry",
        "aqi_feature_store"
    ]
    
    forecast_keywords = ['forecast', 'prediction', 'ml_', 'timeseries', 'ensemble']
    
    print("🔍 Searching for forecast collections in all databases...")
    print("=" * 60)
    
    for db_name in databases:
        try:
            db = client[db_name]
            collections = db.list_collection_names()
            
            forecast_collections = []
            for coll in collections:
                if any(keyword in coll.lower() for keyword in forecast_keywords):
                    forecast_collections.append(coll)
            
            if forecast_collections:
                print(f"\n📊 Database: {db_name}")
                print(f"   Found {len(forecast_collections)} forecast collections:")
                for coll in forecast_collections:
                    count = db[coll].count_documents({})
                    print(f"   - {coll}: {count:,} records")
                    
                    # Show sample
                    if count > 0:
                        sample = db[coll].find_one()
                        print(f"     Sample keys: {list(sample.keys())[:10]}")
            else:
                print(f"\n📭 Database: {db_name} - No forecast collections found")
                
        except Exception as e:
            print(f"\n❌ Error checking database {db_name}: {str(e)}")
    
    client.close()
    print("\n" + "=" * 60)
    print("✅ Forecast check completed")

if __name__ == "__main__":
    main()