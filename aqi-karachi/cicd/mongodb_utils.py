"""
MongoDB utilities for feature store and model registry
"""
import os
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING, DESCENDING
import pandas as pd
import numpy as np
from bson.binary import Binary
import pickle
import json
from typing import Optional, Dict, List, Tuple, Any

class MongoDBManager:
    def __init__(self, uri: str = None):
        """Initialize MongoDB connection with environment variable fallback"""
        # Use provided URI or get from environment
        self.uri = uri or os.getenv("MONGODB_URI")
        if not self.uri:
            raise ValueError("MongoDB URI not provided and MONGODB_URI environment variable not set")
        
        # Get database names from environment with defaults
        self.db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")  # ADDED THIS LINE
        self.feature_store_db = os.getenv("FEATURE_STORE_DB", "aqi_feature_store")
        self.model_registry_db = os.getenv("MODEL_REGISTRY_DB", "aqi_model_registry")
        self.pipeline_logs_db = os.getenv("LOGS_DB", "aqi_pipeline_logs")
        
        # Connection configuration
        connect_timeout = int(os.getenv("MONGODB_CONNECT_TIMEOUT", "5000"))
        socket_timeout = int(os.getenv("MONGODB_SOCKET_TIMEOUT", "30000"))
        
        self.client = MongoClient(
            self.uri,
            connectTimeoutMS=connect_timeout,
            socketTimeoutMS=socket_timeout,
            serverSelectionTimeoutMS=5000
        )
        
        # Test connection
        self.client.admin.command('ping')
        self.setup_indexes()
        
        print(f"✅ Connected to MongoDB")
        print(f"   Main DB: {self.db_name}")  # UPDATED THIS LINE
        print(f"   Feature Store: {self.feature_store_db}")
        print(f"   Model Registry: {self.model_registry_db}")
        print(f"   Pipeline Logs: {self.pipeline_logs_db}")
    
    def setup_indexes(self):
        """Create necessary indexes for performance"""
        # Feature store indexes
        fs_db = self.client[self.feature_store_db]
        fs_db.raw_data.create_index([('timestamp', DESCENDING)])
        fs_db.raw_data.create_index([('location', ASCENDING)])
        fs_db.processed_features.create_index([('version', ASCENDING)])
        fs_db.processed_features.create_index([('timestamp', DESCENDING)])
        fs_db.feature_versions.create_index([('timestamp', DESCENDING)])
        
        # Model registry indexes
        mr_db = self.client[self.model_registry_db]
        mr_db.models.create_index([('is_production', ASCENDING)])
        mr_db.models.create_index([('timestamp', DESCENDING)])
        mr_db.models.create_index([('model_name', ASCENDING)])
        mr_db.training_runs.create_index([('timestamp', DESCENDING)])
        mr_db.evaluations.create_index([('timestamp', DESCENDING)])
        
        # Pipeline logs indexes
        pl_db = self.client[self.pipeline_logs_db]
        pl_db.execution_logs.create_index([('timestamp', DESCENDING)])
        pl_db.execution_logs.create_index([('step', ASCENDING)])
        pl_db.data_versions.create_index([('timestamp', DESCENDING)])
    
    def log_pipeline_step(self, step: str, status: str, details: Dict = None) -> str:
        """Log pipeline execution to MongoDB"""
        db = self.client[self.pipeline_logs_db]
        
        # Get environment info
        run_id = os.getenv('GITHUB_RUN_ID', 'local')
        commit_sha = os.getenv('GITHUB_SHA', 'local')
        event_name = os.getenv('GITHUB_EVENT_NAME', 'manual')
        is_github_actions = os.getenv('GITHUB_ACTIONS', 'false') == 'true'
        
        log_entry = {
            'step': step,
            'status': status,
            'timestamp': datetime.utcnow(),
            'details': details or {},
            'run_id': run_id,
            'commit_sha': commit_sha,
            'event_name': event_name,
            'is_github_actions': is_github_actions,
            'environment': 'production' if is_github_actions else 'development'
        }
        
        result = db.execution_logs.insert_one(log_entry)
        return str(result.inserted_id)
    
    def store_raw_data(self, data: pd.DataFrame, source: str = "api") -> int:
        """Store raw data in MongoDB"""
        db = self.client[self.feature_store_db]
        collection = db['raw_data']
        
        # Convert DataFrame to records
        records = data.to_dict('records')
        
        # Add metadata
        timestamp = datetime.utcnow()
        for record in records:
            record['source'] = source
            record['ingestion_timestamp'] = timestamp
            if 'timestamp' not in record:
                record['timestamp'] = timestamp
            if 'location' not in record:
                record['location'] = os.getenv('CITY_NAME', 'Unknown')
        
        # Insert data with batch size
        batch_size = int(os.getenv('MONGODB_BATCH_SIZE', '100'))
        inserted_count = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            result = collection.insert_many(batch, ordered=False)
            inserted_count += len(result.inserted_ids)
        
        return inserted_count
    
    def get_latest_raw_data(self, limit_hours: int = 24) -> pd.DataFrame:
        """Get latest raw data within time window"""
        db = self.client[self.feature_store_db]
        collection = db['raw_data']
        
        cutoff_time = datetime.utcnow() - timedelta(hours=limit_hours)
        
        cursor = collection.find(
            {'timestamp': {'$gte': cutoff_time}}
        ).sort('timestamp', DESCENDING)
        
        data = list(cursor)
        if not data:
            return pd.DataFrame()
        
        # Remove MongoDB _id field
        for item in data:
            item.pop('_id', None)
        
        return pd.DataFrame(data)
    
    def store_features(self, features_df: pd.DataFrame, version: str) -> Tuple[int, str]:
        """Store engineered features with versioning"""
        db = self.client[self.feature_store_db]
        
        # Store features
        features_collection = db['processed_features']
        
        # Add version and timestamp to each record
        records = features_df.to_dict('records')
        timestamp = datetime.utcnow()
        location = os.getenv('CITY_NAME', 'Unknown')
        
        for record in records:
            record['version'] = version
            record['timestamp'] = timestamp
            record['location'] = location
            record['feature_count'] = len(features_df.columns)
        
        # Insert in batches
        batch_size = int(os.getenv('MONGODB_BATCH_SIZE', '100'))
        inserted_count = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            result = features_collection.insert_many(batch, ordered=False)
            inserted_count += len(result.inserted_ids)
        
        # Update feature version metadata
        version_collection = db['feature_versions']
        version_collection.insert_one({
            'version': version,
            'timestamp': timestamp,
            'location': location,
            'feature_count': len(features_df.columns),
            'record_count': len(features_df),
            'feature_names': list(features_df.columns),
            'data_types': {col: str(dtype) for col, dtype in features_df.dtypes.items()}
        })
        
        return inserted_count, version
    
    def get_latest_features(self, version: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        """Get latest features from MongoDB"""
        db = self.client[self.feature_store_db]
        
        # Get version
        if not version:
            version_doc = db['feature_versions'].find_one(
                sort=[('timestamp', DESCENDING)]
            )
            if not version_doc:
                return pd.DataFrame(), "no_version"
            version = version_doc['version']
        
        # Fetch features for this version
        cursor = db['processed_features'].find(
            {'version': version}
        ).sort('timestamp', ASCENDING)
        
        data = list(cursor)
        if not data:
            return pd.DataFrame(), version
        
        # Remove MongoDB _id and metadata fields
        for item in data:
            item.pop('_id', None)
            item.pop('version', None)
            item.pop('feature_count', None)
            item.pop('location', None)
        
        return pd.DataFrame(data), version
    
    def store_model(self, model: Any, metrics: Dict, model_name: str, 
                    feature_version: str, model_type: str = "sklearn") -> str:
        """Store trained model in MongoDB registry"""
        db = self.client[self.model_registry_db]
        
        # Serialize model
        model_binary = Binary(pickle.dumps(model))
        
        # Get promotion threshold from environment
        promotion_threshold = float(os.getenv('MODEL_PROMOTION_THRESHOLD', '0.75'))
        
        # Create model document
        model_doc = {
            'model_name': model_name,
            'model_data': model_binary,
            'model_type': model_type,
            'feature_version': feature_version,
            'metrics': metrics,
            'timestamp': datetime.utcnow(),
            'location': os.getenv('CITY_NAME', 'Unknown'),
            'is_production': False,
            'promotion_threshold': promotion_threshold,
            'parameters': getattr(model, 'get_params', lambda: {})()
        }
        
        # Insert into models collection
        models_collection = db['models']
        model_id = models_collection.insert_one(model_doc).inserted_id
        
        # Log training run
        runs_collection = db['training_runs']
        runs_collection.insert_one({
            'model_id': model_id,
            'model_name': model_name,
            'feature_version': feature_version,
            'timestamp': datetime.utcnow(),
            'metrics': metrics,
            'model_type': model_type,
            'location': os.getenv('CITY_NAME', 'Unknown')
        })
        
        return str(model_id)
    
    def get_production_model(self, model_name: Optional[str] = None) -> Tuple[Optional[Any], Optional[Dict]]:
        """Get current production model"""
        db = self.client[self.model_registry_db]
        
        query = {'is_production': True}
        if model_name:
            query['model_name'] = model_name
        
        model_doc = db['models'].find_one(query)
        
        if model_doc and 'model_data' in model_doc:
            model = pickle.loads(model_doc['model_data'])
            return model, model_doc
        return None, None
    
    def evaluate_and_promote(self, model_id: str, promotion_threshold: float = None) -> bool:
        """Evaluate model and promote to production if better"""
        db = self.client[self.model_registry_db]
        
        # Use environment threshold if not provided
        if promotion_threshold is None:
            promotion_threshold = float(os.getenv('MODEL_PROMOTION_THRESHOLD', '0.75'))
        
        # Get model
        model_doc = db['models'].find_one({'_id': model_id})
        if not model_doc:
            return False
        
        # Check against current production model
        production_model = db['models'].find_one({
            'is_production': True,
            'model_name': model_doc['model_name']
        })
        
        should_promote = False
        
        if not production_model:
            # No production model exists for this name
            should_promote = True
        else:
            # Compare metrics
            new_score = model_doc['metrics'].get('score', model_doc['metrics'].get('r2', 0))
            prod_score = production_model['metrics'].get('score', production_model['metrics'].get('r2', 0))
            
            # Add other metrics comparison logic here
            improvement = new_score - prod_score
            
            if new_score > promotion_threshold and improvement > 0.01:
                should_promote = True
        
        if should_promote:
            # Demote current production model
            if production_model:
                db['models'].update_one(
                    {'_id': production_model['_id']},
                    {'$set': {'is_production': False}}
                )
            
            # Promote new model
            db['models'].update_one(
                {'_id': model_id},
                {'$set': {'is_production': True}}
            )
            
            # Log promotion
            db['evaluations'].insert_one({
                'model_id': model_id,
                'previous_model_id': production_model['_id'] if production_model else None,
                'action': 'promoted_to_production',
                'timestamp': datetime.utcnow(),
                'metrics': model_doc['metrics'],
                'improvement': new_score - prod_score if production_model else new_score,
                'promotion_threshold': promotion_threshold,
                'location': os.getenv('CITY_NAME', 'Unknown')
            })
        
        return should_promote
    
    def store_prediction(self, features: Dict, prediction: Any, 
                        model_id: str, model_name: str) -> str:
        """Store prediction results"""
        db = self.client[self.model_registry_db]
        
        prediction_doc = {
            'model_id': model_id,
            'model_name': model_name,
            'features': features,
            'prediction': prediction,
            'timestamp': datetime.utcnow(),
            'source': 'pipeline',
            'location': os.getenv('CITY_NAME', 'Unknown')
        }
        
        result = db['predictions'].insert_one(prediction_doc)
        return str(result.inserted_id)
    
    def get_data_statistics(self) -> Dict:
        """Get pipeline statistics"""
        stats = {
            'raw_data_count': self.client[self.feature_store_db]['raw_data'].count_documents({}),
            'feature_versions_count': self.client[self.feature_store_db]['feature_versions'].count_documents({}),
            'latest_feature_version': self.client[self.feature_store_db]['feature_versions'].find_one(
                sort=[('timestamp', DESCENDING)]
            ),
            'models_count': self.client[self.model_registry_db]['models'].count_documents({}),
            'production_models': list(self.client[self.model_registry_db]['models'].find(
                {'is_production': True}
            )),
            'pipeline_logs_count': self.client[self.pipeline_logs_db]['execution_logs'].count_documents({}),
            'location': os.getenv('CITY_NAME', 'Unknown')
        }
        
        return stats
    
    def cleanup_old_data(self, retention_days: int = None):
        """Clean up old data based on retention policy"""
        if retention_days is None:
            retention_days = int(os.getenv('DATA_RETENTION_DAYS', '30'))
        
        cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
        
        # Clean up old raw data
        fs_db = self.client[self.feature_store_db]
        raw_deleted = fs_db['raw_data'].delete_many({
            'timestamp': {'$lt': cutoff_time}
        }).deleted_count
        
        # Clean up old features
        features_deleted = fs_db['processed_features'].delete_many({
            'timestamp': {'$lt': cutoff_time}
        }).deleted_count
        
        # Clean up old logs
        pl_db = self.client[self.pipeline_logs_db]
        logs_deleted = pl_db['execution_logs'].delete_many({
            'timestamp': {'$lt': cutoff_time}
        }).deleted_count
        
        return {
            'raw_data_deleted': raw_deleted,
            'features_deleted': features_deleted,
            'logs_deleted': logs_deleted,
            'retention_days': retention_days,
            'cutoff_time': cutoff_time
        }