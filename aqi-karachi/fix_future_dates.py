# save as: aqi-karachi/fix_future_dates.py
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd

load_dotenv()

def ensure_datetime(timestamp):
    """Convert to datetime safely"""
    try:
        if isinstance(timestamp, datetime):
            return timestamp
        return pd.to_datetime(timestamp)
    except:
        return datetime.now()

uri = os.getenv("MONGODB_URI")
client = MongoClient(uri)
db = client[os.getenv("MONGODB_DATABASE", "aqi_predictor")]

print("🔧 Fixing future dates in MongoDB...")
now = datetime.now()

# Fix all collections with future dates
for coll_name in ['timeseries_forecasts_3day', 'ml_recursive_forecasts', 'ensemble_forecasts_3day', 'aqi_measurements']:
    if coll_name in db.list_collection_names():
        print(f"\nChecking {coll_name}...")
        
        # Fix timestamp field
        docs = list(db[coll_name].find({'timestamp': {'$type': 'string'}}))
        for doc in docs:
            if 'timestamp' in doc:
                dt = ensure_datetime(doc['timestamp'])
                if dt > now + timedelta(days=7):
                    db[coll_name].update_one(
                        {'_id': doc['_id']},
                        {'$set': {'timestamp': now.isoformat()}}
                    )
                    print(f"  Fixed timestamp: {doc['timestamp']} -> {now}")
        
        # Fix created_at field
        docs = list(db[coll_name].find({'created_at': {'$type': 'string'}}))
        for doc in docs:
            if 'created_at' in doc:
                dt = ensure_datetime(doc['created_at'])
                if dt > now + timedelta(days=7):
                    db[coll_name].update_one(
                        {'_id': doc['_id']},
                        {'$set': {'created_at': now.isoformat()}}
                    )
                    print(f"  Fixed created_at: {doc['created_at']} -> {now}")

print("\n✅ All future dates fixed!")
client.close()