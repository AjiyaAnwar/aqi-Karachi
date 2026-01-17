# Create historical data collector
"""
Collect 45 days of historical AQI data immediately
"""
import requests
from datetime import datetime, timedelta
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def collect_45_days():
    """Fetch and store 45 days of historical data"""
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE')]
    collection = db['aqi_measurements']
    
    # Get data for last 45 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=45)
    
    print(f"Collecting data from {start_date.date()} to {end_date.date()}")
    
    # Open-Meteo historical API
    params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "hourly": ["pm2_5", "pm10", "us_aqi"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "Asia/Karachi"
    }
    
    response = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params=params
    )
    
    if response.status_code == 200:
        data = response.json()
        hourly = data.get("hourly", {})
        
        # Process and store
        documents = []
        for i in range(len(hourly.get("time", []))):
            doc = {
                "timestamp": hourly["time"][i],
                "aqi": hourly.get("us_aqi", [])[i],
                "pm2_5": hourly.get("pm2_5", [])[i],
                "pm10": hourly.get("pm10", [])[i],
                "source": "Open-Meteo Historical",
                "location": "Karachi",
                "collection_type": "historical"
            }
            documents.append(doc)
        
        if documents:
            result = collection.insert_many(documents)
            print(f"✅ Inserted {len(result.inserted_ids)} historical records")
        else:
            print("❌ No data received")
    
    client.close()

if __name__ == "__main__":
    collect_45_days()
