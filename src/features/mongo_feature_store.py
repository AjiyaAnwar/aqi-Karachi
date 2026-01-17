"""
MongoDB Feature Store for AQI Prediction
"""
import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from bson import ObjectId
import os
from dotenv import load_dotenv
import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MongoFeatureStore:
    """MongoDB-based feature store and model registry"""
    
    def __init__(self):
        self.uri = os.getenv("MONGODB_URI", "")
        self.db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        self.client = None
        self.db = None
        
        if self.uri:
            self.connect()
        else:
            logger.warning("MongoDB URI not found. Using local file system fallback.")
            self.use_filesystem = True
    
    def connect(self):
        """Connect to MongoDB Atlas"""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            self.db = self.client[self.db_name]
            
            # Test connection
            self.client.server_info()
            logger.info(f"✅ Connected to MongoDB Atlas: {self.db_name}")
            
            # Initialize collections
            self._initialize_collections()
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.use_filesystem = True
    
    def _initialize_collections(self):
        """Initialize MongoDB collections if they don't exist"""
        collections = [
            "raw_aqi_data",
            "hourly_features", 
            "daily_features",
            "models",
            "predictions",
            "metadata"
        ]
        
        existing_collections = self.db.list_collection_names()
        
        for collection in collections:
            if collection not in existing_collections:
                self.db.create_collection(collection)
                logger.info(f"  Created collection: {collection}")
    
    # ===== DATA STORAGE METHODS =====
    
    def save_raw_aqi_data(self, df: pd.DataFrame, source: str = "open_meteo"):
        """Save raw AQI data to MongoDB"""
        try:
            if self.use_filesystem:
                return self._save_to_filesystem(df, "raw_aqi_data")
            
            collection = self.db["raw_aqi_data"]
            
            # Convert DataFrame to records
            records = df.to_dict('records')
            
            # Add metadata
            for record in records:
                record["source"] = source
                record["ingestion_time"] = datetime.now()
                record["city"] = os.getenv("CITY_NAME", "Karachi")
            
            # Insert with deduplication
            inserted_count = 0
            for record in records:
                # Use timestamp as unique key
                query = {
                    "timestamp": record.get("timestamp"),
                    "city": record.get("city"),
                    "source": source
                }
                
                result = collection.update_one(
                    query,
                    {"$set": record},
                    upsert=True
                )
                
                if result.upserted_id:
                    inserted_count += 1
            
            logger.info(f"✅ Saved {inserted_count} raw AQI records to MongoDB")
            return True
            
        except Exception as e:
            logger.error(f"Error saving raw AQI data: {e}")
            return False
    
    def save_hourly_features(self, df: pd.DataFrame, feature_set: str = "v1"):
        """Save hourly features"""
        try:
            if self.use_filesystem:
                return self._save_to_filesystem(df, "hourly_features")
            
            collection = self.db["hourly_features"]
            
            records = df.to_dict('records')
            
            for record in records:
                record["feature_set"] = feature_set
                record["ingestion_time"] = datetime.now()
            
            inserted_count = 0
            for record in records:
                query = {
                    "timestamp": record.get("timestamp"),
                    "feature_set": feature_set
                }
                
                result = collection.update_one(
                    query,
                    {"$set": record},
                    upsert=True
                )
                
                if result.upserted_id:
                    inserted_count += 1
            
            logger.info(f"✅ Saved {inserted_count} hourly features to MongoDB")
            return True
            
        except Exception as e:
            logger.error(f"Error saving hourly features: {e}")
            return False
    
    def save_daily_features(self, df: pd.DataFrame, feature_set: str = "v1"):
        """Save daily aggregated features"""
        try:
            if self.use_filesystem:
                return self._save_to_filesystem(df, "daily_features")
            
            collection = self.db["daily_features"]
            
            records = df.to_dict('records')
            
            for record in records:
                record["feature_set"] = feature_set
                record["ingestion_time"] = datetime.now()
            
            inserted_count = 0
            for record in records:
                query = {
                    "date": record.get("date"),
                    "feature_set": feature_set
                }
                
                result = collection.update_one(
                    query,
                    {"$set": record},
                    upsert=True
                )
                
                if result.upserted_id:
                    inserted_count += 1
            
            logger.info(f"✅ Saved {inserted_count} daily features to MongoDB")
            return True
            
        except Exception as e:
            logger.error(f"Error saving daily features: {e}")
            return False
    
    # ===== DATA RETRIEVAL METHODS =====
    
    def get_training_data(self, days: int = 45, feature_set: str = "v1"):
        """Get training data for model training"""
        try:
            if self.use_filesystem:
                return self._load_from_filesystem("daily_features")
            
            collection = self.db["daily_features"]
            
            # Calculate cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days)).date()
            
            # Query data
            query = {
                "feature_set": feature_set,
                "date": {"$gte": cutoff_date}
            }
            
            cursor = collection.find(query).sort("date", 1)
            data = list(cursor)
            
            if data:
                df = pd.DataFrame(data)
                
                # Remove MongoDB _id field
                if '_id' in df.columns:
                    df = df.drop('_id', axis=1)
                
                # Convert date strings to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                logger.info(f"✅ Retrieved {len(df)} training records from MongoDB")
                return df
            
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
        
        return pd.DataFrame()
    
    # ===== MODEL REGISTRY METHODS =====
    
    def save_model(self, model, model_name: str, version: str, metrics: dict = None):
        """Save model to MongoDB model registry"""
        try:
            if self.use_filesystem:
                return self._save_model_to_filesystem(model, model_name, version, metrics)
            
            collection = self.db["models"]
            
            # Serialize model
            model_bytes = pickle.dumps(model)
            
            model_doc = {
                "model_name": model_name,
                "version": version,
                "model_data": model_bytes,
                "metrics": metrics or {},
                "created_at": datetime.now(),
                "is_production": False
            }
            
            # Save to MongoDB
            result = collection.update_one(
                {"model_name": model_name, "version": version},
                {"$set": model_doc},
                upsert=True
            )
            
            logger.info(f"✅ Model {model_name} v{version} saved to MongoDB")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_name: str, version: str = None):
        """Load model from registry"""
        try:
            if self.use_filesystem:
                return self._load_model_from_filesystem(model_name, version)
            
            collection = self.db["models"]
            
            query = {"model_name": model_name}
            if version:
                query["version"] = version
            else:
                query["is_production"] = True
            
            model_doc = collection.find_one(query)
            
            if model_doc and "model_data" in model_doc:
                model = pickle.loads(model_doc["model_data"])
                logger.info(f"✅ Loaded model {model_name} v{model_doc.get('version')}")
                return model, model_doc
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
        
        return None, None
    
    # ===== FILESYSTEM FALLBACK METHODS =====
    
    def _save_to_filesystem(self, df: pd.DataFrame, collection_name: str):
        """Fallback: Save to filesystem"""
        os.makedirs("data/filesystem", exist_ok=True)
        filename = f"data/filesystem/{collection_name}_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"📁 Saved to filesystem: {filename}")
        return True
    
    def _load_from_filesystem(self, collection_name: str):
        """Fallback: Load from filesystem"""
        import glob
        files = glob.glob(f"data/filesystem/{collection_name}_*.csv")
        
        if files:
            latest_file = max(files, key=os.path.getctime)
            df = pd.read_csv(latest_file)
            logger.info(f"📁 Loaded from filesystem: {latest_file}")
            return df
        
        return pd.DataFrame()
    
    def _save_model_to_filesystem(self, model, model_name: str, version: str, metrics: dict):
        """Fallback: Save model to filesystem"""
        os.makedirs("models", exist_ok=True)
        filename = f"models/{model_name}_v{version}.joblib"
        
        model_data = {
            "model": model,
            "metrics": metrics,
            "created_at": datetime.now()
        }
        
        import joblib
        joblib.dump(model_data, filename)
        logger.info(f"📁 Model saved to filesystem: {filename}")
        return True
    
    def _load_model_from_filesystem(self, model_name: str, version: str = None):
        """Fallback: Load model from filesystem"""
        import glob
        import joblib
        
        if version:
            filename = f"models/{model_name}_v{version}.joblib"
            if os.path.exists(filename):
                data = joblib.load(filename)
                return data.get("model"), data
        else:
            # Get latest version
            pattern = f"models/{model_name}_v*.joblib"
            files = glob.glob(pattern)
            
            if files:
                latest_file = max(files, key=os.path.getctime)
                data = joblib.load(latest_file)
                return data.get("model"), data
        
        return None, None

# Test the feature store
if __name__ == "__main__":
    print("🧪 Testing MongoDB Feature Store...")
    
    store = MongoFeatureStore()
    
    # Create sample data
    sample_data = pd.DataFrame({
        "timestamp": [datetime.now()],
        "pm2_5": [45.2],
        "pm10": [78.5],
        "aqi": [145]
    })
    
    # Test saving
    store.save_raw_aqi_data(sample_data)
    
    # Test loading
    training_data = store.get_training_data(days=7)
    print(f"Training data retrieved: {len(training_data)} records")
    
    print("✅ Feature store test complete!")
