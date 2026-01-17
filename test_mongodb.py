# Run test
"""
Test MongoDB Atlas Connection
"""
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test_connection():
    """Test MongoDB Atlas connection"""
    print("🔗 Testing MongoDB Atlas connection...")
    
    uri = os.getenv("MONGODB_URI", "")
    
    if not uri:
        print("❌ MONGODB_URI not found in .env file")
        print("Please add: MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/")
        return False
    
    try:
        # Connect to MongoDB
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.server_info()
        print("✅ MongoDB Atlas connection successful!")
        
        # List databases
        dbs = client.list_database_names()
        print(f"📊 Available databases: {dbs}")
        
        # Create/use our database
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        db = client[db_name]
        
        # Create a test collection
        test_collection = db["test_connection"]
        test_collection.insert_one({"message": "Connection test", "timestamp": "now"})
        
        # Read it back
        result = test_collection.find_one({"message": "Connection test"})
        print(f"✅ Test document inserted: {result}")
        
        # Clean up
        test_collection.delete_many({})
        
        client.close()
        print("✅ All tests passed!")
        return True
        
    except pymongo.errors.ServerSelectionTimeoutError as e:
        print(f"❌ Connection timeout: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Make sure IP is whitelisted in MongoDB Atlas")
        print("3. Verify username/password in connection string")
        return False
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    test_connection()
