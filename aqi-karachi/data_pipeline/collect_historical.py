"""
Collect 45 days of historical AQI data - UPDATED with retry logic
"""
import requests
from datetime import datetime, timedelta
from pymongo import MongoClient
import os
import time
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def collect_45_days():
    """Fetch and store 45 days of historical data with retry logic"""
    # Get MongoDB connection
    mongodb_uri = os.getenv('MONGODB_URI')
    mongodb_db = os.getenv('MONGODB_DATABASE', 'aqi_predictor')
    
    if not mongodb_uri:
        print("❌ MONGODB_URI not set in .env file")
        return
    
    print("📥 Starting AQI data collection for Karachi...")
    
    # Get data for last 45 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=45)
    
    print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"📍 Location: Karachi (24.8607, 67.0011)")
    
    # Open-Meteo historical API parameters
    params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "hourly": ["pm2_5", "pm10", "us_aqi", "carbon_monoxide", "nitrogen_dioxide"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "Asia/Karachi"
    }
    
    # Try multiple attempts with different endpoints
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
                print(f"   Requesting data...")
                
                # Increase timeout for slow connections
                response = requests.get(
                    endpoint,
                    params=params,
                    timeout=60  # Increased timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Successfully received data from API")
                    break
                else:
                    print(f"⚠️  API returned status code: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"⏱️  Timeout on attempt {attempt + 1}. Retrying in 5 seconds...")
                time.sleep(5)
            except requests.exceptions.ConnectionError:
                print(f"🔌 Connection error on attempt {attempt + 1}. Retrying in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                print(f"⚠️  Error on attempt {attempt + 1}: {type(e).__name__}")
                time.sleep(5)
        
        if data:
            break
    
    if not data:
        print("❌ Failed to fetch data after all attempts")
        
        # Create sample data for development
        print("📝 Creating sample data for development...")
        create_sample_data(mongodb_uri, mongodb_db, start_date, end_date)
        return
    
    # Process the data
    hourly = data.get("hourly", {})
    
    if not hourly or 'time' not in hourly:
        print("❌ No hourly data in response")
        return
    
    times = hourly.get("time", [])
    aqi_values = hourly.get("us_aqi", [])
    pm25_values = hourly.get("pm2_5", [])
    pm10_values = hourly.get("pm10", [])
    
    print(f"📊 Data summary:")
    print(f"   Time points: {len(times)}")
    print(f"   AQI values: {len(aqi_values)}")
    print(f"   PM2.5 values: {len(pm25_values)}")
    print(f"   PM10 values: {len(pm10_values)}")
    
    # Process and store
    documents = []
    for i in range(len(times)):
        doc = {
            "timestamp": times[i],
            "aqi": aqi_values[i] if i < len(aqi_values) else 0,
            "pm25": pm25_values[i] if i < len(pm25_values) else 0,
            "pm10": pm10_values[i] if i < len(pm10_values) else 0,
            "source": "Open-Meteo",
            "location": "Karachi",
            "collected_at": datetime.now().isoformat()
        }
        documents.append(doc)
    
    if documents:
        # Connect to MongoDB
        try:
            client = MongoClient(mongodb_uri)
            db = client[mongodb_db]
            
            # Save to both collections for compatibility
            collections = {
                'aqi_measurements': db['aqi_measurements'],
                'aqi_features': db['aqi_features']
            }
            
            for coll_name, collection in collections.items():
                # Clear old data first
                collection.delete_many({})
                
                # Insert new data
                result = collection.insert_many(documents)
                print(f"✅ Saved {len(result.inserted_ids)} records to {coll_name}")
            
            # Calculate statistics
            aqi_stats = [doc['aqi'] for doc in documents if doc['aqi'] > 0]
            if aqi_stats:
                avg_aqi = sum(aqi_stats) / len(aqi_stats)
                max_aqi = max(aqi_stats)
                min_aqi = min(aqi_stats)
                
                print(f"\n📈 AQI Statistics:")
                print(f"   Average AQI: {avg_aqi:.1f}")
                print(f"   Maximum AQI: {max_aqi:.1f}")
                print(f"   Minimum AQI: {min_aqi:.1f}")
                
                # Categorize average AQI
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
            
            # Save sample to CSV
            os.makedirs('data', exist_ok=True)
            df = pd.DataFrame(documents[:100])  # First 100 records
            csv_path = 'data/sample_data.csv'
            df.to_csv(csv_path, index=False)
            print(f"📁 Sample data saved to {csv_path}")
            
            client.close()
            
        except Exception as e:
            print(f"❌ MongoDB error: {e}")
    else:
        print("❌ No documents to save")

def create_sample_data(mongodb_uri, mongodb_db, start_date, end_date):
    """Create sample data for development when API fails"""
    try:
        client = MongoClient(mongodb_uri)
        db = client[mongodb_db]
        
        # Generate sample data
        documents = []
        current_time = start_date
        
        # Create 45 days of hourly data
        hours = 45 * 24  # 45 days * 24 hours
        
        for i in range(hours):
            # Simulate realistic AQI patterns (higher during day, lower at night)
            hour = current_time.hour
            if 8 <= hour <= 20:  # Daytime
                base_aqi = 80 + (i % 30)  # 80-110 during day
            else:  # Nighttime
                base_aqi = 50 + (i % 20)  # 50-70 at night
            
            # Add some variation
            aqi = base_aqi + (i % 15)
            
            doc = {
                "timestamp": current_time.isoformat(),
                "aqi": aqi,
                "pm25": aqi * 0.4,  # Rough conversion
                "pm10": aqi * 0.6,   # Rough conversion
                "source": "Sample",
                "location": "Karachi",
                "collected_at": datetime.now().isoformat(),
                "is_sample": True
            }
            documents.append(doc)
            current_time += timedelta(hours=1)
        
        # Save to both collections
        collections = {
            'aqi_measurements': db['aqi_measurements'],
            'aqi_features': db['aqi_features']
        }
        
        for coll_name, collection in collections.items():
            collection.delete_many({})
            result = collection.insert_many(documents[:500])  # Limit to 500 samples
            print(f"📝 Created {len(result.inserted_ids)} sample records in {coll_name}")
        
        # Save to CSV
        os.makedirs('data', exist_ok=True)
        df = pd.DataFrame(documents[:100])
        df.to_csv('data/sample_data.csv', index=False)
        print(f"📁 Sample CSV saved to data/sample_data.csv")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("AQI Karachi - Data Collection Script")
    print("=" * 50)
    
    collect_45_days()
    
    print("\n" + "=" * 50)
    print("Data collection process completed!")
    print("=" * 50)
    
    # Keep terminal open
    input("\nPress Enter to exit...")