"""
Incremental Data Fetch for AQI Karachi
Fetches only new data since last update (last 3-6 hours)
"""
import requests
from datetime import datetime, timedelta
from pymongo import MongoClient
import os
import time
from dotenv import load_dotenv
import pandas as pd
import sys

load_dotenv()

def fetch_incremental_data(hours_back=6):
    """Fetch only new data since last update"""
    # Get MongoDB connection
    mongodb_uri = os.getenv('MONGODB_URI')
    mongodb_db = os.getenv('MONGODB_DATABASE', 'aqi_predictor')
    
    if not mongodb_uri:
        print("❌ MONGODB_URI not set in .env file")
        return
    
    print("📥 Starting INCREMENTAL AQI data collection for Karachi...")
    
    # Connect to MongoDB to get last timestamp
    client = MongoClient(mongodb_uri)
    db = client[mongodb_db]
    
    # Check last record timestamp
    last_record = db.aqi_measurements.find_one(
        sort=[('timestamp', -1)]
    )
    
    if last_record:
        # Start from last timestamp (minus 1 hour to avoid gaps)
        last_timestamp = last_record.get('timestamp')
        if isinstance(last_timestamp, str):
            start_date = datetime.fromisoformat(last_timestamp.replace('Z', '')) - timedelta(hours=1)
        else:
            start_date = last_timestamp - timedelta(hours=1)
        print(f"📅 Fetching from last record: {start_date}")
    else:
        # No data yet, fetch last 6 hours
        start_date = datetime.now() - timedelta(hours=hours_back)
        print(f"📅 No existing data, fetching last {hours_back} hours")
    
    # End date is now
    end_date = datetime.now()
    
    # Ensure we don't fetch too far back
    max_days_back = 3  # Don't fetch more than 3 days
    if (end_date - start_date).days > max_days_back:
        start_date = end_date - timedelta(days=max_days_back)
        print(f"⚠️  Limiting fetch to last {max_days_back} days")
    
    print(f"📍 Location: Karachi (24.8607, 67.0011)")
    print(f"📅 Fetch range: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
    
    # Open-Meteo API parameters
    params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "hourly": ["pm2_5", "pm10", "us_aqi", "carbon_monoxide", "nitrogen_dioxide"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "Asia/Karachi"
    }
    
    # Try multiple endpoints
    endpoints = [
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        "https://air-quality-api.open-meteo.com/v1/forecast"
    ]
    
    max_retries = 3
    data = None
    
    for attempt in range(max_retries):
        for endpoint in endpoints:
            try:
                print(f"\n🔄 Attempt {attempt + 1}/{max_retries} - Using endpoint: {endpoint}")
                
                response = requests.get(
                    endpoint,
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Successfully received incremental data")
                    break
                else:
                    print(f"⚠️  API returned status code: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"⏱️  Timeout on attempt {attempt + 1}. Retrying...")
                time.sleep(2)
            except Exception as e:
                print(f"⚠️  Error on attempt {attempt + 1}: {type(e).__name__}")
                time.sleep(2)
        
        if data:
            break
    
    if not data:
        print("❌ Failed to fetch incremental data")
        
        # Check if we have enough historical data
        count = db.aqi_measurements.count_documents({})
        if count > 100:
            print(f"✅ Using existing {count} records in database")
            client.close()
            return True
        else:
            print("⚠️  Not enough data, consider running full historical collection")
            client.close()
            return False
    
    # Process the data
    hourly = data.get("hourly", {})
    
    if not hourly or 'time' not in hourly:
        print("❌ No hourly data in response")
        client.close()
        return False
    
    times = hourly.get("time", [])
    aqi_values = hourly.get("us_aqi", [])
    pm25_values = hourly.get("pm2_5", [])
    pm10_values = hourly.get("pm10", [])
    
    print(f"📊 New data points: {len(times)}")
    
    # Process and store only new records
    new_documents = []
    existing_timestamps = set()
    
    # Get existing timestamps to avoid duplicates
    if last_record:
        # Get timestamps from last 24 hours to check
        recent_records = db.aqi_measurements.find(
            {'timestamp': {'$gte': start_date.isoformat()}},
            {'timestamp': 1}
        )
        existing_timestamps = {rec['timestamp'] for rec in recent_records}
    
    for i in range(len(times)):
        timestamp_str = times[i]
        
        # Skip if already exists
        if timestamp_str in existing_timestamps:
            continue
            
        doc = {
            "timestamp": timestamp_str,
            "aqi": aqi_values[i] if i < len(aqi_values) else 0,
            "pm25": pm25_values[i] if i < len(pm25_values) else 0,
            "pm10": pm10_values[i] if i < len(pm10_values) else 0,
            "source": "Open-Meteo",
            "location": "Karachi",
            "collected_at": datetime.now().isoformat(),
            "fetch_type": "incremental"
        }
        new_documents.append(doc)
    
    if new_documents:
        # Insert only new documents
        result = db.aqi_measurements.insert_many(new_documents)
        print(f"✅ Added {len(result.inserted_ids)} new records")
        
        # Also update features collection
        
        print(f"✅ Updated features collection with {len(features_result.inserted_ids)} records")
        
        # Calculate statistics
        new_aqi_stats = [doc['aqi'] for doc in new_documents if doc['aqi'] > 0]
        if new_aqi_stats:
            avg_aqi = sum(new_aqi_stats) / len(new_aqi_stats)
            print(f"\n📈 New AQI Statistics:")
            print(f"   Average: {avg_aqi:.1f}")
            print(f"   Range: {min(new_aqi_stats):.1f} to {max(new_aqi_stats):.1f}")
            
            # Categorize
            if avg_aqi <= 50:
                category = "Good 😊"
            elif avg_aqi <= 100:
                category = "Moderate 😐"
            elif avg_aqi <= 150:
                category = "Unhealthy for Sensitive Groups 😷"
            elif avg_aqi <= 200:
                category = "Unhealthy 🤒"
            elif avg_aqi <= 300:
                category = "Very Unhealthy 🏥"
            else:
                category = "Hazardous ☣️"
            
            print(f"   Air Quality: {category}")
    else:
        print("ℹ️  No new data to add (all timestamps already exist)")
    
    # Update data age stats
    total_count = db.aqi_measurements.count_documents({})
    oldest = db.aqi_measurements.find_one(sort=[('timestamp', 1)])
    newest = db.aqi_measurements.find_one(sort=[('timestamp', -1)])
    
    if oldest and newest:
        print(f"\n📊 Database Summary:")
        print(f"   Total records: {total_count:,}")
        print(f"   Oldest data: {oldest['timestamp']}")
        print(f"   Newest data: {newest['timestamp']}")
    
    client.close()
    print("\n✅ Incremental data collection complete!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🔄 AQI Karachi - Incremental Data Collection")
    print("=" * 60)
    
    success = fetch_incremental_data(hours_back=6)
    
    print("\n" + "=" * 60)
    print("✅ Process completed!" if success else "⚠️ Process completed with issues")
    print("=" * 60)