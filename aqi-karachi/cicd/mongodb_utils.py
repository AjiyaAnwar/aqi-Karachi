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
import re
import hashlib
from dataclasses import dataclass
from enum import Enum

class ModelStatus(Enum):
    """Model deployment status"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

@dataclass
class ModelMetadata:
    """Metadata for model versioning"""
    model_name: str
    version: str
    status: ModelStatus
    feature_version: str
    performance: Dict[str, float]
    created_at: datetime

class MongoDBManager:
    def __init__(self, uri: str = None):
        """Initialize MongoDB connection with environment variable fallback"""
        # Use provided URI or get from environment
        self.uri = uri or os.getenv("MONGODB_URI")
        if not self.uri:
            raise ValueError("MongoDB URI not provided and MONGODB_URI environment variable not set")
        
        # Get database names from environment with defaults
        self.db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
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
        print(f"   Main DB: {self.db_name}")
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
        fs_db.feature_versions.create_index([('version', ASCENDING)], unique=True)
        fs_db.feature_statistics.create_index([('feature_version', ASCENDING)])
        
        # Model registry indexes
        mr_db = self.client[self.model_registry_db]
        mr_db.models.create_index([('status', ASCENDING)])
        mr_db.models.create_index([('timestamp', DESCENDING)])
        mr_db.models.create_index([('model_name', ASCENDING)])
        mr_db.models.create_index([('model_name', ASCENDING), ('version', ASCENDING)], unique=True)
        mr_db.training_runs.create_index([('timestamp', DESCENDING)])
        mr_db.evaluations.create_index([('timestamp', DESCENDING)])
        mr_db.predictions.create_index([('timestamp', DESCENDING)])
        mr_db.predictions.create_index([('model_version', ASCENDING)])
        mr_db.model_metadata.create_index([('model_name', ASCENDING), ('version', ASCENDING)], unique=True)
        
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
    
    # ========== FEATURE STORE METHODS ==========
    
    def get_next_feature_version(self, major_increment: bool = False) -> str:
        """Get next feature version (v1.0, v1.1, etc.)"""
        db = self.client[self.feature_store_db]
        
        # Get latest version
        latest = db['feature_versions'].find_one(
            sort=[('timestamp', DESCENDING)]
        )
        
        if not latest:
            return "v1.0"
        
        current_version = latest.get('version', 'v1.0')
        
        # Parse version string like "v1.2"
        match = re.match(r'v(\d+)\.(\d+)', current_version)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            
            if major_increment:
                return f"v{major + 1}.0"
            else:
                return f"v{major}.{minor + 1}"
        else:
            return "v1.0"
    
    def calculate_feature_hash(self, features_df: pd.DataFrame) -> str:
        """Calculate hash of feature set for change detection"""
        # Sort columns and convert to string for consistent hashing
        sorted_columns = sorted(features_df.columns)
        feature_string = "|".join(sorted_columns)
        data_hash = hashlib.md5(features_df.values.tobytes()).hexdigest()[:8]
        
        return f"{feature_string[:50]}_{data_hash}"
    
    def store_features_with_versioning(self, features_df: pd.DataFrame, 
                                      version: str = None,
                                      description: str = "",
                                      is_training: bool = True) -> Tuple[str, str]:
        """Store engineered features with automatic versioning"""
        db = self.client[self.feature_store_db]
        
        # Generate version if not provided
        if version is None:
            feature_hash = self.calculate_feature_hash(features_df)
            base_version = self.get_next_feature_version()
            version = f"{base_version}_{feature_hash}"
        
        # Store features
        features_collection = db['processed_features']
        
        # Add version and timestamp to each record
        records = features_df.to_dict('records')
        timestamp = datetime.utcnow()
        location = os.getenv('CITY_NAME', 'Unknown')
        
        for record in records:
            record['feature_version'] = version
            record['ingestion_timestamp'] = timestamp
            record['location'] = location
            record['is_training'] = is_training
            record['feature_hash'] = self.calculate_feature_hash(pd.DataFrame([record]))
        
        # Insert in batches
        batch_size = int(os.getenv('MONGODB_BATCH_SIZE', '100'))
        inserted_count = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            result = features_collection.insert_many(batch, ordered=False)
            inserted_count += len(result.inserted_ids)
        
        # Calculate statistics
        stats = {
            'mean': features_df.mean().to_dict(),
            'std': features_df.std().to_dict(),
            'min': features_df.min().to_dict(),
            'max': features_df.max().to_dict(),
            'count': len(features_df),
            'null_counts': features_df.isnull().sum().to_dict()
        }
        
        # Update feature version metadata
        version_collection = db['feature_versions']
        version_doc = {
            'version': version,
            'timestamp': timestamp,
            'location': location,
            'description': description,
            'feature_count': len(features_df.columns),
            'record_count': len(features_df),
            'feature_names': list(features_df.columns),
            'data_types': {col: str(dtype) for col, dtype in features_df.dtypes.items()},
            'feature_hash': self.calculate_feature_hash(features_df),
            'is_training': is_training,
            'statistics': stats
        }
        
        version_collection.update_one(
            {'version': version},
            {'$set': version_doc},
            upsert=True
        )
        
        # Store statistics separately
        stats_collection = db['feature_statistics']
        for feature_name, value in stats['mean'].items():
            stats_collection.update_one(
                {'feature_version': version, 'feature_name': feature_name},
                {'$set': {
                    'mean': value,
                    'std': stats['std'].get(feature_name),
                    'min': stats['min'].get(feature_name),
                    'max': stats['max'].get(feature_name),
                    'timestamp': timestamp
                }},
                upsert=True
            )
        
        print(f"✅ Stored features version {version} with {inserted_count} records")
        return version, str(version_doc.get('_id'))
    
    def get_features_by_version(self, version: str, 
                               start_date: datetime = None,
                               end_date: datetime = None) -> pd.DataFrame:
        """Get features for specific version with date filtering"""
        db = self.client[self.feature_store_db]
        
        query = {'feature_version': version}
        
        if start_date:
            query['ingestion_timestamp'] = {'$gte': start_date}
        if end_date:
            if 'ingestion_timestamp' in query:
                query['ingestion_timestamp']['$lte'] = end_date
            else:
                query['ingestion_timestamp'] = {'$lte': end_date}
        
        cursor = db['processed_features'].find(query).sort('ingestion_timestamp', ASCENDING)
        
        data = list(cursor)
        if not data:
            return pd.DataFrame()
        
        # Convert to DataFrame and remove metadata
        df = pd.DataFrame(data)
        
        # Remove MongoDB specific fields
        columns_to_drop = ['_id', 'feature_version', 'ingestion_timestamp', 
                          'location', 'feature_hash', 'is_training']
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        return df.drop(columns=columns_to_drop)
    
    def list_feature_versions(self, limit: int = 10) -> List[Dict]:
        """List available feature versions"""
        db = self.client[self.feature_store_db]
        
        versions = db['feature_versions'].find(
            {},
            {'version': 1, 'timestamp': 1, 'feature_count': 1, 
             'record_count': 1, 'description': 1, '_id': 0}
        ).sort('timestamp', DESCENDING).limit(limit)
        
        return list(versions)
    
    # ========== MODEL REGISTRY METHODS ==========
    
    def get_next_model_version(self, model_name: str) -> str:
        """Get next model version for a specific model"""
        db = self.client[self.model_registry_db]
        
        # Get latest version for this model
        latest = db['models'].find_one(
            {'model_name': model_name},
            sort=[('timestamp', DESCENDING)]
        )
        
        if not latest:
            return f"{model_name}_v1.0"
        
        current_version = latest.get('version', f"{model_name}_v1.0")
        
        # Parse version string like "xgboost_v1.2"
        match = re.search(r'v(\d+)\.(\d+)$', current_version)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            return f"{model_name}_v{major}.{minor + 1}"
        else:
            return f"{model_name}_v1.0"
    
    def register_model(self, model: Any, model_name: str, 
                      metrics: Dict, feature_version: str,
                      hyperparams: Dict = None,
                      tags: List[str] = None) -> ModelMetadata:
        """Register a new model version with metadata"""
        db = self.client[self.model_registry_db]
        
        # Generate version
        version = self.get_next_model_version(model_name)
        
        # Serialize model
        model_binary = Binary(pickle.dumps(model))
        
        # Create model document
        model_doc = {
            'model_name': model_name,
            'version': version,
            'model_data': model_binary,
            'model_type': type(model).__name__,
            'feature_version': feature_version,
            'hyperparameters': hyperparams or {},
            'metrics': metrics,
            'timestamp': datetime.utcnow(),
            'location': os.getenv('CITY_NAME', 'Unknown'),
            'status': ModelStatus.DEVELOPMENT.value,
            'tags': tags or [],
            'git_commit': os.getenv('GITHUB_SHA', 'local'),
            'run_id': os.getenv('GITHUB_RUN_ID', 'local')
        }
        
        # Insert model
        models_collection = db['models']
        model_id = models_collection.insert_one(model_doc).inserted_id
        
        # Log training run
        runs_collection = db['training_runs']
        runs_collection.insert_one({
            'model_id': model_id,
            'model_name': model_name,
            'version': version,
            'feature_version': feature_version,
            'timestamp': datetime.utcnow(),
            'metrics': metrics,
            'hyperparameters': hyperparams or {},
            'duration_seconds': metrics.get('training_time', 0),
            'location': os.getenv('CITY_NAME', 'Unknown')
        })
        
        # Store metadata separately for fast queries
        metadata_collection = db['model_metadata']
        metadata = ModelMetadata(
            model_name=model_name,
            version=version,
            status=ModelStatus.DEVELOPMENT,
            feature_version=feature_version,
            performance=metrics,
            created_at=datetime.utcnow()
        )
        
        metadata_collection.insert_one({
            'model_name': model_name,
            'version': version,
            'status': metadata.status.value,
            'feature_version': feature_version,
            'performance': metrics,
            'created_at': metadata.created_at,
            'primary_metric': metrics.get('r2', metrics.get('accuracy', 0)),
            'tags': tags or []
        })
        
        print(f"✅ Registered model {model_name} version {version}")
        return metadata
    
    def promote_model(self, model_name: str, version: str, 
                     target_status: ModelStatus = ModelStatus.PRODUCTION) -> bool:
        """Promote/demote a model version"""
        db = self.client[self.model_registry_db]
        
        # Find the model
        model = db['models'].find_one({
            'model_name': model_name,
            'version': version
        })
        
        if not model:
            print(f"❌ Model {model_name} version {version} not found")
            return False
        
        current_status = model.get('status', ModelStatus.DEVELOPMENT.value)
        
        # If promoting to production, demote current production model
        if target_status == ModelStatus.PRODUCTION:
            # Demote current production model (if any)
            db['models'].update_many(
                {
                    'model_name': model_name,
                    'status': ModelStatus.PRODUCTION.value
                },
                {'$set': {'status': ModelStatus.ARCHIVED.value}}
            )
            
            # Also update metadata
            db['model_metadata'].update_many(
                {
                    'model_name': model_name,
                    'status': ModelStatus.PRODUCTION.value
                },
                {'$set': {'status': ModelStatus.ARCHIVED.value}}
            )
        
        # Promote the specified model
        result = db['models'].update_one(
            {'_id': model['_id']},
            {'$set': {'status': target_status.value}}
        )
        
        # Update metadata
        db['model_metadata'].update_one(
            {
                'model_name': model_name,
                'version': version
            },
            {'$set': {'status': target_status.value}}
        )
        
        # Log the promotion
        db['evaluations'].insert_one({
            'model_name': model_name,
            'version': version,
            'from_status': current_status,
            'to_status': target_status.value,
            'timestamp': datetime.utcnow(),
            'reason': 'manual_promotion' if target_status != ModelStatus.ARCHIVED else 'auto_archive',
            'location': os.getenv('CITY_NAME', 'Unknown')
        })
        
        print(f"✅ Promoted {model_name} {version} from {current_status} to {target_status.value}")
        return result.modified_count > 0
    
    def get_model(self, model_name: str, version: str = None, 
                 status: ModelStatus = None) -> Tuple[Any, ModelMetadata]:
        """Retrieve a model and its metadata"""
        db = self.client[self.model_registry_db]
        
        query = {'model_name': model_name}
        
        if version:
            query['version'] = version
        if status:
            query['status'] = status.value
        
        # Get the model document
        model_doc = db['models'].find_one(query, sort=[('timestamp', DESCENDING)])
        
        if not model_doc or 'model_data' not in model_doc:
            return None, None
        
        # Deserialize model
        model = pickle.loads(model_doc['model_data'])
        
        # Get metadata
        metadata_doc = db['model_metadata'].find_one({
            'model_name': model_name,
            'version': model_doc['version']
        })
        
        if metadata_doc:
            metadata = ModelMetadata(
                model_name=metadata_doc['model_name'],
                version=metadata_doc['version'],
                status=ModelStatus(metadata_doc['status']),
                feature_version=metadata_doc['feature_version'],
                performance=metadata_doc['performance'],
                created_at=metadata_doc['created_at']
            )
        else:
            metadata = ModelMetadata(
                model_name=model_doc['model_name'],
                version=model_doc['version'],
                status=ModelStatus(model_doc.get('status', ModelStatus.DEVELOPMENT.value)),
                feature_version=model_doc['feature_version'],
                performance=model_doc.get('metrics', {}),
                created_at=model_doc['timestamp']
            )
        
        return model, metadata
    
    def get_production_models(self) -> List[ModelMetadata]:
        """Get all models currently in production"""
        db = self.client[self.model_registry_db]
        
        metadata_docs = db['model_metadata'].find(
            {'status': ModelStatus.PRODUCTION.value}
        )
        
        models = []
        for doc in metadata_docs:
            models.append(ModelMetadata(
                model_name=doc['model_name'],
                version=doc['version'],
                status=ModelStatus(doc['status']),
                feature_version=doc['feature_version'],
                performance=doc['performance'],
                created_at=doc['created_at']
            ))
        
        return models
    
    def list_model_versions(self, model_name: str = None, 
                          limit: int = 20) -> List[Dict]:
        """List available model versions"""
        db = self.client[self.model_registry_db]
        
        query = {}
        if model_name:
            query['model_name'] = model_name
        
        models = db['model_metadata'].find(
            query,
            {'model_name': 1, 'version': 1, 'status': 1, 
             'primary_metric': 1, 'created_at': 1, 'feature_version': 1, '_id': 0}
        ).sort('created_at', DESCENDING).limit(limit)
        
        return list(models)
    
    def log_prediction(self, model_name: str, version: str,
                      features: Dict, prediction: Any,
                      actual: Any = None) -> str:
        """Log prediction for monitoring and A/B testing"""
        db = self.client[self.model_registry_db]
        
        prediction_doc = {
            'model_name': model_name,
            'model_version': version,
            'features': features,
            'prediction': prediction,
            'actual': actual,
            'timestamp': datetime.utcnow(),
            'location': os.getenv('CITY_NAME', 'Unknown'),
            'error': None if actual is None else float(abs(prediction - actual))
        }
        
        result = db['predictions'].insert_one(prediction_doc)
        
        # Update model metrics if actual is provided
        if actual is not None:
            # Store in evaluations for performance tracking
            db['evaluations'].insert_one({
                'model_name': model_name,
                'version': version,
                'timestamp': datetime.utcnow(),
                'prediction_error': float(abs(prediction - actual)),
                'features_hash': hashlib.md5(str(features).encode()).hexdigest()[:8]
            })
        
        return str(result.inserted_id)
    
    def get_model_performance(self, model_name: str, version: str,
                            hours_back: int = 24) -> Dict:
        """Get recent performance metrics for a model"""
        db = self.client[self.model_registry_db]
        
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        
        # Get recent predictions with actual values
        predictions = list(db['predictions'].find({
            'model_name': model_name,
            'model_version': version,
            'timestamp': {'$gte': cutoff},
            'actual': {'$ne': None}
        }))
        
        if not predictions:
            return {'error_count': 0, 'mae': 0, 'rmse': 0}
        
        errors = [p['error'] for p in predictions if p['error'] is not None]
        
        if not errors:
            return {'error_count': 0, 'mae': 0, 'rmse': 0}
        
        return {
            'error_count': len(errors),
            'mae': float(np.mean(errors)),
            'rmse': float(np.sqrt(np.mean([e**2 for e in errors]))),
            'max_error': float(np.max(errors)),
            'min_error': float(np.min(errors)),
            'last_updated': datetime.utcnow()
        }
    
    # ========== EXISTING METHODS (KEPT FOR BACKWARD COMPATIBILITY) ==========
    
    def store_raw_data(self, data: pd.DataFrame, source: str = "api") -> int:
        """Store raw data in MongoDB (existing method)"""
        # ... keep existing implementation ...
        db = self.client[self.feature_store_db]
        collection = db['raw_data']
        
        records = data.to_dict('records')
        timestamp = datetime.utcnow()
        for record in records:
            record['source'] = source
            record['ingestion_timestamp'] = timestamp
            if 'timestamp' not in record:
                record['timestamp'] = timestamp
            if 'location' not in record:
                record['location'] = os.getenv('CITY_NAME', 'Unknown')
        
        batch_size = int(os.getenv('MONGODB_BATCH_SIZE', '100'))
        inserted_count = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            result = collection.insert_many(batch, ordered=False)
            inserted_count += len(result.inserted_ids)
        
        return inserted_count
    
    def store_features(self, features_df: pd.DataFrame, version: str) -> Tuple[int, str]:
        """Backward compatibility wrapper"""
        version, _ = self.store_features_with_versioning(features_df, version)
        return len(features_df), version
    
    def get_latest_features(self, version: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        """Get latest features (backward compatible)"""
        if version is None:
            versions = self.list_feature_versions(limit=1)
            if not versions:
                return pd.DataFrame(), "no_version"
            version = versions[0]['version']
        
        df = self.get_features_by_version(version)
        return df, version
    
    def store_model(self, model: Any, metrics: Dict, model_name: str, 
                    feature_version: str, model_type: str = "sklearn") -> str:
        """Backward compatibility wrapper"""
        metadata = self.register_model(model, model_name, metrics, feature_version)
        return metadata.version
    
    def get_production_model(self, model_name: Optional[str] = None) -> Tuple[Optional[Any], Optional[Dict]]:
        """Backward compatibility wrapper"""
        if model_name:
            model, metadata = self.get_model(model_name, status=ModelStatus.PRODUCTION)
        else:
            prod_models = self.get_production_models()
            if not prod_models:
                return None, None
            model, metadata = self.get_model(prod_models[0].model_name, 
                                           prod_models[0].version)
        
        if model and metadata:
            return model, {
                'model_name': metadata.model_name,
                'version': metadata.version,
                'performance': metadata.performance,
                'feature_version': metadata.feature_version
            }
        return None, None
    
    def evaluate_and_promote(self, model_id: str, promotion_threshold: float = None) -> bool:
        """Backward compatibility - note: model_id not used in new system"""
        print("⚠️  Using deprecated evaluate_and_promote. Use promote_model instead.")
        return False
    
    def store_prediction(self, features: Dict, prediction: Any, 
                        model_id: str, model_name: str) -> str:
        """Backward compatibility wrapper"""
        return self.log_prediction(model_name, "unknown", features, prediction)
    
    # ========== UTILITY METHODS ==========
    
    def get_data_statistics(self) -> Dict:
        """Get pipeline statistics"""
        stats = {
            'raw_data_count': self.client[self.feature_store_db]['raw_data'].count_documents({}),
            'feature_versions': self.list_feature_versions(limit=5),
            'production_models': [m.__dict__ for m in self.get_production_models()],
            'total_models': self.client[self.model_registry_db]['models'].count_documents({}),
            'recent_predictions': self.client[self.model_registry_db]['predictions'].count_documents({
                'timestamp': {'$gte': datetime.utcnow() - timedelta(hours=24)}
            }),
            'pipeline_logs_count': self.client[self.pipeline_logs_db]['execution_logs'].count_documents({}),
            'location': os.getenv('CITY_NAME', 'Unknown'),
            'timestamp': datetime.utcnow()
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
        
        # Clean up old features (keep only latest 3 versions)
        versions_to_keep = self.list_feature_versions(limit=3)
        versions_to_keep = [v['version'] for v in versions_to_keep]
        
        features_deleted = fs_db['processed_features'].delete_many({
            'feature_version': {'$nin': versions_to_keep}
        }).deleted_count
        
        # Clean up old logs
        pl_db = self.client[self.pipeline_logs_db]
        logs_deleted = pl_db['execution_logs'].delete_many({
            'timestamp': {'$lt': cutoff_time}
        }).deleted_count
        
        # Archive old models (not in production)
        mr_db = self.client[self.model_registry_db]
        models_archived = mr_db['models'].update_many(
            {
                'timestamp': {'$lt': cutoff_time},
                'status': {'$ne': ModelStatus.PRODUCTION.value}
            },
            {'$set': {'status': ModelStatus.ARCHIVED.value}}
        ).modified_count
        
        return {
            'raw_data_deleted': raw_deleted,
            'features_deleted': features_deleted,
            'logs_deleted': logs_deleted,
            'models_archived': models_archived,
            'retention_days': retention_days,
            'cutoff_time': cutoff_time
        }

# Example usage
if __name__ == "__main__":
    # Test the enhanced MongoDB manager
    mongo_manager = MongoDBManager()
    
    # Test feature versioning
    print("\n📊 Testing feature versioning...")
    versions = mongo_manager.list_feature_versions(limit=3)
    print(f"Available feature versions: {versions}")
    
    # Test model registry
    print("\n🤖 Testing model registry...")
    model_versions = mongo_manager.list_model_versions()
    print(f"Available model versions: {model_versions}")
    
    # Test production models
    prod_models = mongo_manager.get_production_models()
    print(f"Production models: {[m.model_name for m in prod_models]}")
    
    # Get statistics
    stats = mongo_manager.get_data_statistics()
    print(f"\n📈 Pipeline statistics: {stats}")
