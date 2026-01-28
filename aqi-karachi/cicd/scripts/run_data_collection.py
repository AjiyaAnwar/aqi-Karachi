#!/usr/bin/env python
"""
Data Collection Runner - INTEGRATED WITH YOUR CODE
Runs your actual collect_historical.py script
"""
import os
import sys
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

# ==================== ENVIRONMENT SETUP ====================
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded environment from: {env_path}")
else:
    load_dotenv()

# Import MongoDB utilities
try:
    from cicd.mongodb_utils import MongoDBManager
except ImportError:
    sys.path.append(str(project_root / 'cicd'))
    from mongodb_utils import MongoDBManager

def run_your_data_collection():
    """Run your actual data collection script"""
    print("📥 Running your data collection script...")
    
    script_path = project_root / 'data_pipeline' / 'collect_historical.py'
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return None
    
    try:
        # Run your script using subprocess to capture output
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=600  # 10 minute timeout
        )
        
        print("📊 Data collection output:")
        print("-" * 40)
        
        # Print last 20 lines of output
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[-20:]:  # Show last 20 lines
                if line.strip():
                    print(f"  {line.strip()}")
        
        print("-" * 40)
        
        if result.returncode == 0:
            print("✅ Your data collection script completed successfully")
            
            # Extract number of records from output if possible
            records_count = 0
            for line in lines:
                if 'records' in line.lower():
                    import re
                    match = re.search(r'(\d+)\s+records', line)
                    if match:
                        records_count = int(match.group(1))
                        break
            
            return {
                'success': True,
                'records_count': records_count,
                'output_lines': len(lines)
            }
        else:
            print(f"❌ Your data collection script failed with code {result.returncode}")
            if result.stderr:
                print("Error output:")
                for line in result.stderr.strip().split('\n')[-10:]:
                    if line.strip():
                        print(f"  ❌ {line.strip()}")
            
            return {
                'success': False,
                'error_code': result.returncode,
                'error_message': result.stderr[:200] if result.stderr else "Unknown error"
            }
            
    except subprocess.TimeoutExpired:
        print("❌ Data collection timed out after 5 minutes")
        return {
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"❌ Error running data collection: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Main data collection function - integrates with your code"""
    print("=" * 60)
    print("📥 AQI DATA COLLECTION PIPELINE")
    print("📍 City: Karachi")
    print("=" * 60)
    
    # Get MongoDB URI from environment
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        print("❌ MONGODB_URI environment variable not set")
        print("   Please set MONGODB_URI in your .env file")
        sys.exit(1)
    
    # Initialize MongoDB manager
    try:
        mongo_manager = MongoDBManager(mongodb_uri)
        print(f"✅ Connected to MongoDB: {mongo_manager.feature_store_db}")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {str(e)}")
        sys.exit(1)
    
    # Log start
    log_id = mongo_manager.log_pipeline_step('data_collection', 'started', {
        'city': 'Karachi',
        'run_type': 'github_actions' if os.getenv('GITHUB_ACTIONS') == 'true' else 'local',
        'script': 'collect_historical.py'
    })
    
    try:
        # Step 1: Run YOUR data collection script
        print("\n1️⃣ RUNNING DATA COLLECTION SCRIPT")
        print("-" * 40)
        
        result = run_your_data_collection()
        
        if not result or not result.get('success', False):
            print("❌ Data collection failed")
            mongo_manager.log_pipeline_step('data_collection', 'failed', {
                'error': result.get('error', 'Unknown error') if result else 'No result',
                'parent_log_id': log_id
            })
            return
        
        # Step 2: Get statistics from MongoDB
        print("\n2️⃣ COLLECTING DATABASE STATISTICS")
        print("-" * 40)
        
        stats = mongo_manager.get_data_statistics()
        
        print(f"   📈 Total raw data records: {stats['raw_data_count']:,}")
        print(f"   🏷️  Feature versions: {stats['feature_versions_count']:,}")
        
        if stats['latest_feature_version']:
            version = stats['latest_feature_version']['version']
            timestamp = stats['latest_feature_version']['timestamp']
            print(f"   🔄 Latest feature version: {version}")
            print(f"   🕒 Created: {timestamp}")
        
        # Check aqi_measurements collection specifically
        db = mongo_manager.client[mongo_manager.feature_store_db]
        if 'aqi_measurements' in db.list_collection_names():
            aqi_count = db['aqi_measurements'].count_documents({})
            print(f"   📊 AQI measurements: {aqi_count:,}")
            
            # Get latest AQI
            latest_aqi = db['aqi_measurements'].find_one(sort=[('timestamp', -1)])
            if latest_aqi:
                aqi_value = latest_aqi.get('aqi', latest_aqi.get('us_aqi', 0))
                timestamp = latest_aqi.get('timestamp', 'Unknown')
                print(f"   📍 Latest AQI: {aqi_value}")
                print(f"   🕒 Latest timestamp: {timestamp}")
        
        # Step 3: Log success
        print("\n3️⃣ LOGGING EXECUTION")
        print("-" * 40)
        
        mongo_manager.log_pipeline_step('data_collection', 'completed', {
            'success': True,
            'records_count': stats['raw_data_count'],
            'feature_versions': stats['feature_versions_count'],
            'latest_feature_version': stats['latest_feature_version']['version'] if stats['latest_feature_version'] else None,
            'parent_log_id': log_id,
            'city': 'Karachi'
        })
        
        print("\n" + "=" * 60)
        print("✅ DATA COLLECTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Data collection pipeline failed: {str(e)}")
        traceback.print_exc()
        
        mongo_manager.log_pipeline_step('data_collection', 'failed', {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'parent_log_id': log_id
        })
        
        print("\n" + "=" * 60)
        print("❌ PIPELINE FAILED - Check error above")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    # Print environment info
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Project Root: {project_root}")
    print(f"🌍 Environment: {'GitHub Actions' if os.getenv('GITHUB_ACTIONS') == 'true' else 'Local'}")
    
    # Run main function
    main()