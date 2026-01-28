# test_data.py
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")

client = MongoClient(uri)
db = client[db_name]

# Check collections
collections = db.list_collection_names()
print(f"Available collections: {collections}")

# Check aqi_measurements
print(f"\naqi_measurements count: {db.aqi_measurements.count_documents({})}")
if db.aqi_measurements.count_documents({}) > 0:
    latest = db.aqi_measurements.find_one(sort=[('timestamp', -1)])
    print(f"Latest in aqi_measurements: {latest}")

# Check aqi_features
print(f"\naqi_features count: {db.aqi_features.count_documents({})}")
if db.aqi_features.count_documents({}) > 0:
    latest = db.aqi_features.find_one(sort=[('timestamp', -1)])
    print(f"Latest in aqi_features: {latest}")

client.close()