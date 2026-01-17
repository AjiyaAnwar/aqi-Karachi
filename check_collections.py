
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('MONGODB_DATABASE')]

print("📊 Database Status:")
print(f"Database: {db.name}")
print(f"Collections: {db.list_collection_names()}")

if "aqi_measurements" in db.list_collection_names():
    collection = db["aqi_measurements"]
    count = collection.count_documents({})
    print(f"\n✅ 'aqi_measurements' collection exists!")
    print(f"Total documents: {count}")
    
    # Show latest document
    latest = collection.find_one(sort=[("timestamp", -1)])
    if latest:
        print("\n📄 Latest AQI measurement:")
        print(f"  AQI: {latest.get('aqi', 'N/A')}")
        print(f"  PM2.5: {latest.get('pm2_5', 'N/A')}")
        print(f"  Time: {latest.get('timestamp', 'N/A')}")
    
    # Show sample data
    print("\n📋 Sample of stored data:")
    for doc in collection.find().limit(3):
        print(f"  - AQI {doc.get('aqi')} at {doc.get('timestamp', '')[:16]}")
else:
    print("\n❌ 'aqi_measurements' collection not created yet")

client.close()
