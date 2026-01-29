#!/usr/bin/env python
"""
Data Collection Runner - UPDATED for incremental mode
"""
import os
import sys
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

def run_incremental_data_collection():
    """Run incremental data collection"""
    print("📥 Running incremental data collection...")
    
    script_path = project_root / 'data_pipeline' / 'collect_historical.py'
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return None
    
    try:
        # Run with incremental flag
        result = subprocess.run(
            [sys.executable, str(script_path), '--incremental', '--hours=6'],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300
        )
        
        print("Output:")
        print(result.stdout[-500:] if result.stdout else "No output")
        
        if result.returncode == 0:
            print("✅ Incremental data collection completed")
            return {'success': True, 'type': 'incremental'}
        else:
            print(f"❌ Failed with code {result.returncode}")
            return {'success': False, 'error': result.stderr[:200]}
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {'success': False, 'error': str(e)}

def run_full_data_collection():
    """Run full data collection (only if needed)"""
    print("📚 Running full data collection...")
    
    script_path = project_root / 'data_pipeline' / 'collect_historical.py'
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=600
        )
        
        print("Output:")
        print(result.stdout[-500:] if result.stdout else "No output")
        
        if result.returncode == 0:
            print("✅ Full data collection completed")
            return {'success': True, 'type': 'full'}
        else:
            print(f"❌ Failed with code {result.returncode}")
            return {'success': False, 'error': result.stderr[:200]}
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {'success': False, 'error': str(e)}

def main():
    print("=" * 60)
    print("📥 AQI DATA COLLECTION RUNNER")
    print("=" * 60)
    
    # Check if we should run full or incremental
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        print("❌ MONGODB_URI not set")
        sys.exit(1)
    
    try:
        from pymongo import MongoClient
        client = MongoClient(mongodb_uri)
        db = client[os.getenv("MONGODB_DATABASE", "aqi_predictor")]
        
        # Check existing data
        count = db.aqi_measurements.count_documents({})
        
        if count < 100:
            print(f"⚠️  Low data count ({count}), running full collection")
            result = run_full_data_collection()
        else:
            print(f"✅ Existing data: {count} records, running incremental")
            result = run_incremental_data_collection()
        
        client.close()
        
        if result and result.get('success'):
            print("\n✅ Data collection completed successfully!")
        else:
            print("\n❌ Data collection failed")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()