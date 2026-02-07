# save as: aqi-karachi/fix_future_timestamps.py
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd

load_dotenv()

uri = os.getenv("MONGODB_URI")
client = MongoClient(uri)
db = client[os.getenv("MONGODB_DATABASE", "aqi_predictor")]

print("🔧 PERMANENTLY fixing future timestamps in MongoDB...")
now = datetime.now()
print(f"Current time: {now}")

# Fix ALL forecast collections
collections = ['timeseries_forecasts_3day', 'ml_recursive_forecasts', 'ensemble_forecasts_3day']

for coll_name in collections:
    if coll_name in db.list_collection_names():
        print(f"\n📁 Fixing {coll_name}...")
        
        # Get ALL documents
        docs = list(db[coll_name].find({}))
        print(f"  Found {len(docs)} documents")
        
        fixed = 0
        for doc in docs:
            updates = {}
            
            # Check and fix timestamp fields
            for field in ['timestamp', 'created_at', 'forecast_time', 'prediction_time']:
                if field in doc:
                    try:
                        # Parse the value
                        if isinstance(doc[field], str):
                            dt = pd.to_datetime(doc[field])
                        elif isinstance(doc[field], datetime):
                            dt = doc[field]
                        else:
                            continue
                        
                        # If date is in future (more than 2 days), fix it
                        if dt > now + timedelta(days=2):
                            updates[field] = now
                            fixed += 1
                            print(f"    Fixed {field}: {dt} -> {now}")
                    
                    except Exception as e:
                        print(f"    Error parsing {field}: {e}")
            
            # Apply updates if any
            if updates:
                db[coll_name].update_one(
                    {'_id': doc['_id']},
                    {'$set': updates}
                )
        
        print(f"  ✅ Fixed {fixed} fields in {coll_name}")

print("\n🎉 ALL future timestamps fixed!")
print("\nNext steps:")
print("1. RESTART your dashboard")
print("2. Click 'Update Predictions' again")
print("3. Should show 'Updated X.Xh ago' (positive number)")
client.close()