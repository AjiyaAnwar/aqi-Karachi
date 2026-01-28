#!/usr/bin/env python
"""
Log execution results to MongoDB
"""
import os
import sys
import argparse
from datetime import datetime
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
    parser = argparse.ArgumentParser(description='Log pipeline execution to MongoDB')
    parser.add_argument('--run-id', required=True, help='GitHub Run ID or local run ID')
    parser.add_argument('--status', required=True, help='Job status (started, completed, failed)')
    parser.add_argument('--trigger', required=True, help='Trigger event (push, schedule, manual)')
    parser.add_argument('--commit', required=True, help='Commit SHA')
    parser.add_argument('--step', default='pipeline', help='Pipeline step name')
    parser.add_argument('--details', default='{}', help='Additional details as JSON string')
    
    args = parser.parse_args()
    
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        print("❌ MONGODB_URI not set")
        sys.exit(1)
    
    try:
        import json
        details = json.loads(args.details)
    except:
        details = {'raw_details': args.details}
    
    try:
        mongo_manager = MongoDBManager(mongodb_uri)
        
        log_id = mongo_manager.log_pipeline_step(
            step=args.step,
            status=args.status,
            details={
                'run_id': args.run_id,
                'trigger': args.trigger,
                'commit_sha': args.commit,
                'timestamp': datetime.utcnow(),
                **details
            }
        )
        
        print(f"✅ Logged execution: {args.step} - {args.status} (ID: {log_id})")
        
    except Exception as e:
        print(f"❌ Failed to log execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()