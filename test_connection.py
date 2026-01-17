from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv('MONGODB_URI')
print(f"Testing connection with: mongodb+srv://user:user@cluster0.pj50bgz.mongodb.net/...")

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("✅ SUCCESS: Connected to MongoDB!")
    
    # Test database access
    db = client["aqi_predictor"]
    print(f"✅ Database 'aqi_predictor' accessible")
    
    # List collections
    collections = db.list_collection_names()
    print(f"Collections: {collections}")
    
    client.close()
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("\nCommon fixes:")
    print("1. Check MongoDB Atlas → Network Access → Add your IP")
    print("2. Verify cluster is 'Active' not 'Inactive'")
    print("3. Check if database user 'user' exists in Atlas")
