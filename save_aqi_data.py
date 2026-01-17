"""
Continuous AQI Data Collector for MongoDB
Runs every hour to save AQI measurements to feature store
"""
import schedule
import time
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import requests
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

def fetch_current_aqi():
    """Fetch current AQI from Open-Meteo"""
    try:
        params = {
            "latitude": float(os.getenv("CITY_LAT", 24.8607)),
            "longitude": float(os.getenv("CITY_LON", 67.0011)),
            "current": ["us_aqi", "pm2_5", "pm10", "us_aqi_pm2_5", "us_aqi_pm10"],
            "timezone": os.getenv("TIMEZONE", "Asia/Karachi")
        }
        
        response = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            current = data.get("current", {})
            
            if current:
                return {
                    "aqi": current.get("us_aqi", 0),
                    "pm2_5": current.get("pm2_5", 0),
                    "pm10": current.get("pm10", 0),
                    "aqi_pm25": current.get("us_aqi_pm2_5", 0),
                    "aqi_pm10": current.get("us_aqi_pm10", 0),
                    "timestamp": datetime.now().isoformat(),
                    "source": "Open-Meteo",
                    "location": os.getenv("CITY_NAME", "Karachi"),
                    "latitude": float(os.getenv("CITY_LAT", 24.8607)),
                    "longitude": float(os.getenv("CITY_LON", 67.0011))
                }
    except Exception as e:
        logger.error(f"Error fetching AQI: {e}")
    
    return None

def save_current_aqi():
    """Save current AQI to MongoDB feature store"""
    try:
        # Get MongoDB connection
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client[os.getenv("MONGODB_DATABASE")]
        collection = db["aqi_measurements"]
        
        # Fetch current AQI
        aqi_data = fetch_current_aqi()
        
        if aqi_data:
            # Add metadata
            aqi_data["stored_at"] = datetime.now().isoformat()
            aqi_data["collection_type"] = "hourly"
            aqi_data["batch_id"] = f"batch_{datetime.now().strftime('%Y%m%d_%H')}"
            
            # Insert to MongoDB
            result = collection.insert_one(aqi_data)
            
            logger.info(f"✅ Saved AQI {aqi_data['aqi']} at {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"   Document ID: {result.inserted_id}")
            
            return result.inserted_id
        else:
            logger.warning("⚠️ No AQI data received")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error saving to MongoDB: {e}")
        return None

def create_indexes():
    """Create MongoDB indexes for better query performance"""
    try:
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client[os.getenv("MONGODB_DATABASE")]
        collection = db["aqi_measurements"]
        
        # Create indexes
        collection.create_index([("timestamp", 1)])  # For time-based queries
        collection.create_index([("location", 1)])   # For location queries
        collection.create_index([("aqi", 1)])        # For AQI range queries
        
        logger.info("✅ Created MongoDB indexes")
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")

def main():
    """Main function to run the data collector"""
    logger.info("🚀 Starting AQI Data Collector Service")
    logger.info(f"📍 Location: {os.getenv('CITY_NAME', 'Karachi')}")
    logger.info(f"📊 Interval: Every {os.getenv('COLLECTION_INTERVAL_HOURS', 1)} hour(s)")
    
    # Create indexes on startup
    create_indexes()
    
    # Run immediately on startup
    logger.info("🔄 Running initial data collection...")
    save_current_aqi()
    
    # Schedule regular collection
    interval_hours = int(os.getenv("COLLECTION_INTERVAL_HOURS", 1))
    schedule.every(interval_hours).hours.do(save_current_aqi)
    
    logger.info(f"⏰ Scheduled to run every {interval_hours} hour(s)")
    
    # Keep the script running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("👋 Stopping AQI Data Collector")
        client = MongoClient(os.getenv("MONGODB_URI"))
        client.close()

if __name__ == "__main__":
    # Check environment variables
    required_vars = ["MONGODB_URI", "MONGODB_DATABASE"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        logger.error("Please check your .env file")
        exit(1)
    
    main()
