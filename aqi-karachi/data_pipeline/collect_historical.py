"""
Collect AQI Data for Karachi - UPDATED with incremental mode
Run without parameters for initial 45-day load
Run with --incremental for last 6 hours only
"""
import requests
from datetime import datetime, timedelta
from pymongo import MongoClient
import os
import time
from dotenv import load_dotenv
import pandas as pd
import argparse

load_dotenv()

def collect_aqi_data(incremental=False, hours_back=6):
    """Main data collection function with incremental option"""
    # Get MongoDB connection
    mongodb_uri = os.getenv('MONGODB_URI')
    mongodb_db = os.getenv('MONGODB_DATABASE', 'aqi_predictor')
    
    if not mongodb_uri:
        print("❌ MONGODB_URI not set in .env file")
        return
    
    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    db = client[mongodb_db]
    
    if incremental:
        print("📥 INCREMENTAL MODE: Fetching recent AQI data...")
        
        # Get last timestamp
        last_record = db.aqi_measurements.find_one(sort=[('timestamp', -1)])
        
        if last_record:
            last_timestamp = last_record.get('timestamp')
            if isinstance(last_timestamp, str):
                start_date = datetime.fromisoformat(last_timestamp.replace('Z', '')) - timedelta(hours=1)
            else:
                start_date = last_timestamp - timedelta(hours=1)
            print(f"📅 Fetching from last record: {start_date}")
        else:
            # No data yet, fetch last X hours
            start_date = datetime.now() - timedelta(hours=hours_back)
            print(f"📅 No existing data, fetching last {hours_back} hours")
        
        end_date = datetime.now()
        
    else:
        print("📥 FULL MODE: Fetching 45 days of historical AQI data...")
        
        # Get data for last 45 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=45)
        
        # Clear only if database is empty or very small
        existing_count = db.aqi_measurements.count_documents({})
        if existing_count < 100:
            db.aqi_measurements.delete_many({})
            db.aqi_features.delete_many({})
            print("🗑️  Cleared existing collections")
    
    print(f"📍 Location: Karachi (24.8607, 67.0011)")
    print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
    
    # Open-Meteo API parameters
    params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "hourly": ["pm2_5", "pm10", "us_aqi"],
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
                    print(f"✅ Successfully received data")
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
        print("❌ Failed to fetch data from API")
        
        # Check if we have enough data already
        count = db.aqi_measurements.count_documents({})
        if count > 100:
            print(f"✅ Using existing {count} records in database")
            client.close()
            return True
        else:
            print("⚠️  Creating sample data for development...")
            client.close()
            create_sample_data(mongodb_uri, mongodb_db, start_date, end_date)
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
    
    print(f"📊 Data points received: {len(times)}")
    
    # Process documents
    documents = []
    for i in range(len(times)):
        doc = {
            "timestamp": times[i],
            "aqi": aqi_values[i] if i < len(aqi_values) else 0,
            "pm25": pm25_values[i] if i < len(pm25_values) else 0,
            "pm10": pm10_values[i] if i < len(pm10_values) else 0,
            "source": "Open-Meteo",
            "location": "Karachi",
            "collected_at": datetime.now().isoformat(),
            "fetch_type": "incremental" if incremental else "full"
        }
        documents.append(doc)
    
    if incremental:
        # For incremental mode, check for duplicates
        new_documents = []
        existing_timestamps = set()
        
        # Get recent timestamps to avoid duplicates
        recent_cutoff = datetime.now() - timedelta(days=2)
        recent_records = db.aqi_measurements.find(
            {'timestamp': {'$gte': recent_cutoff.isoformat()}},
            {'timestamp': 1}
        )
        existing_timestamps = {rec['timestamp'] for rec in recent_records}
        
        for doc in documents:
            if doc['timestamp'] not in existing_timestamps:
                new_documents.append(doc)
        
        documents = new_documents
        print(f"📊 After deduplication: {len(documents)} new records")
    
    if documents:
        # Insert in batches
        batch_size = 1000
        total_inserted = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            result = db.aqi_measurements.insert_many(batch)
            # Also update features collection
            db.aqi_features.insert_many(batch)
            total_inserted += len(result.inserted_ids)
            print(f"📦 Inserted batch {i//batch_size + 1}: {len(batch)} records")
        
        print(f"\n✅ {'Incremental' if incremental else 'Full'} data collection complete")
        print(f"📊 Total added: {total_inserted} records")
        
        # Calculate statistics
        aqi_stats = [doc['aqi'] for doc in documents if doc['aqi'] > 0]
        if aqi_stats:
            avg_aqi = sum(aqi_stats) / len(aqi_stats)
            print(f"\n📈 New AQI Statistics:")
            print(f"   Average: {avg_aqi:.1f}")
            print(f"   Range: {min(aqi_stats):.1f} to {max(aqi_stats):.1f}")
            
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
        
        # Update summary
        total_count = db.aqi_measurements.count_documents({})
        oldest = db.aqi_measurements.find_one(sort=[('timestamp', 1)])
        newest = db.aqi_measurements.find_one(sort=[('timestamp', -1)])
        
        print(f"\n📊 Database Summary:")
        print(f"   Total records: {total_count:,}")
        if oldest and newest:
            print(f"   Date range: {oldest['timestamp']} to {newest['timestamp']}")
        
        # Save sample CSV
        os.makedirs('data', exist_ok=True)
        df = pd.DataFrame(documents[:100])
        filename = 'data/incremental_sample.csv' if incremental else 'data/full_sample.csv'
        df.to_csv(filename, index=False)
        print(f"📁 Sample saved to {filename}")
    else:
        print("ℹ️  No new data to add")
    
    client.close()
    return True

def create_sample_data(mongodb_uri, mongodb_db, start_date, end_date):
    """Create sample data if API fails"""
    print("📝 Creating sample data for development...")
    
    client = MongoClient(mongodb_uri)
    db = client[mongodb_db]
    
    # Clear only if empty
    if db.aqi_measurements.count_documents({}) < 10:
        db.aqi_measurements.delete_many({})
        db.aqi_features.delete_many({})
    
    # Generate sample data
    documents = []
    current_time = start_date
    
    # Create data
    hours = int((end_date - start_date).total_seconds() / 3600)
    hours = min(hours, 1000)  # Limit to 1000 records
    
    for i in range(hours):
        # Realistic patterns
        hour = current_time.hour
        if 8 <= hour <= 20:  # Daytime
            base_aqi = 80 + (i % 30)
        else:  # Nighttime
            base_aqi = 50 + (i % 20)
        
        aqi = base_aqi + (i % 15)
        
        doc = {
            "timestamp": current_time.isoformat(),
            "aqi": aqi,
            "pm25": aqi * 0.4,
            "pm10": aqi * 0.6,
            "source": "Sample",
            "location": "Karachi",
            "collected_at": datetime.now().isoformat(),
            "is_sample": True
        }
        documents.append(doc)
        current_time += timedelta(hours=1)
    
    # Insert
    result = db.aqi_measurements.insert_many(documents)
    db.aqi_features.insert_many(documents)
    
    print(f"📝 Created {len(result.inserted_ids)} sample records")
    
    # Save CSV
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(documents[:100])
    df.to_csv('data/sample_data.csv', index=False)
    print("📁 Sample CSV saved to data/sample_data.csv")
    
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect AQI data for Karachi')
    parser.add_argument('--incremental', action='store_true', 
                       help='Fetch only incremental data (last 6 hours)')
    parser.add_argument('--hours', type=int, default=6,
                       help='Hours to fetch in incremental mode (default: 6)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    if args.incremental:
        print("🔄 AQI Karachi - Incremental Data Collection")
    else:
        print("📚 AQI Karachi - Full Historical Data Collection")
    print("=" * 60)
    
    success = collect_aqi_data(
        incremental=args.incremental,
        hours_back=args.hours
    )
    
    print("\n" + "=" * 60)
    print("✅ Process completed!" if success else "⚠️ Process completed with issues")
    print("=" * 60)