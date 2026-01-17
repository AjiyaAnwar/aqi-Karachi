from pymongo import MongoClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('MONGODB_DATABASE')]
collection = db['aqi_measurements']

# Count total documents
total = collection.count_documents({})
print(f"📊 TOTAL RECORDS: {total}")

if total > 0:
    # Get date range
    earliest = collection.find_one(sort=[("timestamp", 1)])
    latest = collection.find_one(sort=[("timestamp", -1)])
    
    print(f"\n📅 DATE RANGE:")
    print(f"  Earliest: {earliest.get('timestamp', 'N/A')}")
    print(f"  Latest:   {latest.get('timestamp', 'N/A')}")
    
    # Count by source
    pipeline = [
        {"$group": {
            "_id": "$source",
            "count": {"$sum": 1}
        }}
    ]
    
    sources = list(collection.aggregate(pipeline))
    print(f"\n🔍 DATA SOURCES:")
    for src in sources:
        print(f"  {src['_id']}: {src['count']} records")
    
    # Check data quality
    print(f"\n✅ DATA QUALITY CHECK:")
    
    # Check for null AQI values
    null_aqi = collection.count_documents({"aqi": None})
    print(f"  Records with null AQI: {null_aqi}")
    
    # Get basic stats
    pipeline = [
        {"$group": {
            "_id": None,
            "avg_aqi": {"$avg": "$aqi"},
            "min_aqi": {"$min": "$aqi"},
            "max_aqi": {"$max": "$aqi"},
            "avg_pm25": {"$avg": "$pm2_5"},
            "avg_pm10": {"$avg": "$pm10"}
        }}
    ]
    
    stats = list(collection.aggregate(pipeline))
    if stats:
        s = stats[0]
        print(f"\n📈 BASIC STATISTICS:")
        print(f"  AQI - Avg: {s.get('avg_aqi', 0):.1f}, Min: {s.get('min_aqi', 0)}, Max: {s.get('max_aqi', 0)}")
        print(f"  PM2.5 Avg: {s.get('avg_pm25', 0):.1f} µg/m³")
        print(f"  PM10 Avg:  {s.get('avg_pm10', 0):.1f} µg/m³")
    
    # Check if we have 45 days of data
    if earliest and latest:
        from dateutil import parser
        try:
            early_date = parser.parse(earliest['timestamp'])
            late_date = parser.parse(latest['timestamp'])
            days_diff = (late_date - early_date).days
            
            print(f"\n⏰ TIME COVERAGE:")
            print(f"  {days_diff} days of data")
            
            if days_diff >= 45:
                print("  ✅ SUFFICIENT for model training!")
            else:
                print(f"  ⚠️ Need {45 - days_diff} more days of data")
        except:
            print("  ⚠️ Could not parse dates")
    
    # Show sample records
    print(f"\n🔍 SAMPLE RECORDS (latest 3):")
    for doc in collection.find().sort("timestamp", -1).limit(3):
        print(f"  {doc.get('timestamp')} - AQI: {doc.get('aqi')}, PM2.5: {doc.get('pm2_5')}")

else:
    print("❌ No data found in collection!")

client.close()

