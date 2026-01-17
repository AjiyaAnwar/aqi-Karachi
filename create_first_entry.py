from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('MONGODB_DATABASE')]

# Create sample data
sample_aqi = {
    "aqi": 145,
    "pm2_5": 45.2,
    "pm10": 78.5,
    "timestamp": datetime.now().isoformat(),
    "source": "Manual Entry",
    "location": "Karachi",
    "latitude": 24.8607,
    "longitude": 67.0011,
    "stored_at": datetime.now().isoformat()
}

result = db.aqi_measurements.insert_one(sample_aqi)
print(f"✅ Created first document with ID: {result.inserted_id}")
print(f"Collection count: {db.aqi_measurements.count_documents({})}")

client.close()
