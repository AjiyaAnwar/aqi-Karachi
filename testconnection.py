# test_connection.py
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def test_connection():
    try:
        # Get URI from .env
        uri = os.getenv('MONGODB_URI')
        print(f"🔗 Connecting to: {uri.split('@')[1] if '@' in uri else uri}")  # Don't print password
        
        # Connect with timeout
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.server_info()
        print("✅ MongoDB Atlas Connected Successfully!")
        
        # List databases
        db_name = os.getenv('MONGODB_DATABASE', 'aqi_predictor')
        db = client[db_name]
        print(f"📊 Using database: {db.name}")
        
        # List collections
        collections = db.list_collection_names()
        print(f"📁 Collections ({len(collections)}):")
        for coll in collections:
            count = db[coll].count_documents({})
            print(f"   • {coll}: {count} documents")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection Failed: {str(e)[:100]}...")
        print("\n🔧 Troubleshooting:")
        print("1. Check your .env file has correct MongoDB Atlas URI")
        print("2. Make sure IP is whitelisted in MongoDB Atlas")
        print("3. Check internet connection")
        print("4. Verify username/password are correct")
        return False

if __name__ == "__main__":
    test_connection()