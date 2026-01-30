#!/usr/bin/env python
"""
Check if new data is available for training
"""
import os
import sys
from datetime import datetime, timedelta
import json
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

try:
    from cicd.mongodb_utils import MongoDBManager
except ImportError:
    sys.path.append(str(project_root / 'cicd'))
    from mongodb_utils import MongoDBManager

def main():
    """Check if we should train new models"""
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        result = {
            'should_train': False, 
            'reason': 'no_mongodb_uri',
            'timestamp': datetime.utcnow().isoformat()
        }
        print(json.dumps(result))
        sys.exit(0)
    
    try:
        mongo_manager = MongoDBManager(mongodb_uri)
    except Exception as e:
        result = {
            'should_train': False, 
            'reason': f'mongodb_connection_failed: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }
        print(json.dumps(result))
        sys.exit(0)
    
    # Check for new data in last 6 hours
    cutoff_time = datetime.utcnow() - timedelta(hours=6)
    
    db = mongo_manager.client[mongo_manager.feature_store_db]
    
    # Check raw data
    new_raw_data = db['aqi_measurements'].count_documents({
    'timestamp': {'$gte': cutoff_time}
    })
    
    # Check feature versions
    latest_feature_version_doc = db['feature_versions'].find_one(
        sort=[('timestamp', -1)]
    )
    
    should_train = False
    reason = ""
    
    if new_raw_data > 0:
        should_train = True
        reason = f"New raw data available: {new_raw_data} records"
    elif latest_feature_version_doc:
        # Check if we've trained on this version
        feature_version = latest_feature_version_doc['version']
        
        model_db = mongo_manager.client[mongo_manager.model_registry_db]
        trained_on_version = model_db['models'].count_documents({
            'feature_version': feature_version
        })
        
        if trained_on_version == 0:
            should_train = True
            reason = f"No models trained on latest feature version: {feature_version}"
        else:
            reason = f"Already trained on latest feature version: {feature_version}"
    else:
        reason = "No feature versions available"
    
    result = {
        'should_train': should_train,
        'reason': reason,
        'new_raw_data': new_raw_data,
        'latest_feature_version': latest_feature_version_doc['version'] if latest_feature_version_doc else None,
        'latest_feature_timestamp': latest_feature_version_doc['timestamp'].isoformat() if latest_feature_version_doc else None,
        'cutoff_time': cutoff_time.isoformat(),
        'check_timestamp': datetime.utcnow().isoformat(),
        'city': os.getenv('CITY_NAME', 'Unknown')
    }
    
    print(json.dumps(result, default=str))
    
    # Also write to file for GitHub Actions outputs
    with open('data_status.json', 'w') as f:
        json.dump(result, f, default=str)

if __name__ == "__main__":
    from pymongo import DESCENDING
    main()