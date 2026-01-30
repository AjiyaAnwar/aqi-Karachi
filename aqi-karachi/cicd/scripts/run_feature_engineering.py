#!/usr/bin/env python
"""
Feature Engineering Runner - INTEGRATED WITH YOUR CODE
Runs your actual features.py script
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

# ==================== HELPER FUNCTIONS ====================
def get_raw_data_collection(db_manager):
    """Find raw data in MongoDB - checks both databases"""
    
    # First check in the main database
    main_db = db_manager.client[os.getenv("MONGODB_DATABASE", "aqi_predictor")]
    if 'aqi_measurements' in main_db.list_collection_names():
        count = main_db['aqi_measurements'].count_documents({})
        if count > 0:
            print(f"✅ Found {count:,} raw data records in main database")
            return main_db, count
    
    # Then check in feature store database
    feature_db = db_manager.client[db_manager.feature_store_db]
    if 'aqi_measurements' in feature_db.list_collection_names():
        count = feature_db['aqi_measurements'].count_documents({})
        if count > 0:
            print(f"✅ Found {count:,} raw data records in feature store database")
            return feature_db, count
    
    # Also check 'raw_data' collection (as found by EDA)
    if 'raw_data' in main_db.list_collection_names():
        count = main_db['raw_data'].count_documents({})
        if count > 0:
            print(f"✅ Found {count:,} raw data records in 'raw_data' collection")
            return main_db, count
    
    return None, 0

def run_your_feature_engineering():
    """Run your actual feature engineering script"""
    print("🔧 Running your feature engineering script...")
    
    script_path = project_root / 'data_pipeline' / 'features.py'
    
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
            timeout=300  # 5 minute timeout
        )
        
        print("📊 Feature engineering output:")
        print("-" * 40)
        
        # Print relevant output
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            
            # Filter and show important lines
            important_lines = []
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in [
                    'step', 'feature', 'shape', 'record', 'saved', 'error', 
                    'warning', 'created', 'final', 'columns', 'sample'
                ]):
                    important_lines.append(line.strip())
            
            # Show all important lines (up to 30)
            for line in important_lines[:30]:
                if line:
                    print(f"  {line}")
            
            # Show count of lines processed
            print(f"  ... processed {len(lines)} lines total")
        
        print("-" * 40)
        
        if result.returncode == 0:
            print("✅ Your feature engineering script completed successfully")
            
            # Extract feature information from output
            feature_count = 0
            record_count = 0
            for line in lines:
                if 'feature' in line.lower() and 'shape' in line.lower():
                    import re
                    match = re.search(r'\((\d+),\s*(\d+)\)', line)
                    if match:
                        record_count = int(match.group(1))
                        feature_count = int(match.group(2))
                        break
            
            return {
                'success': True,
                'feature_count': feature_count,
                'record_count': record_count,
                'output_lines': len(lines)
            }
        else:
            print(f"❌ Your feature engineering script failed with code {result.returncode}")
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
        print("❌ Feature engineering timed out after 5 minutes")
        return {
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"❌ Error running feature engineering: {str(e)}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Main feature engineering function - integrates with your code"""
    print("=" * 60)
    print("🔧 FEATURE ENGINEERING PIPELINE")
    print("📍 City: Karachi")
    print("=" * 60)
    
    # Get MongoDB URI from environment
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        print("❌ MONGODB_URI environment variable not set")
        sys.exit(1)
    
    # Initialize MongoDB manager
    try:
        mongo_manager = MongoDBManager(mongodb_uri)
        print(f"✅ Connected to MongoDB")
        print(f"   Main DB: {os.getenv('MONGODB_DATABASE', 'aqi_predictor')}")
        print(f"   Feature Store: {mongo_manager.feature_store_db}")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {str(e)}")
        sys.exit(1)
    
    # Log start
    log_id = mongo_manager.log_pipeline_step('feature_engineering', 'started', {
        'city': 'Karachi',
        'run_type': 'github_actions' if os.getenv('GITHUB_ACTIONS') == 'true' else 'local',
        'script': 'features.py'
    })
    
    try:
        # Step 1: Check if raw data exists
        print("\n1️⃣ CHECKING FOR RAW DATA")
        print("-" * 40)
        
        raw_db, raw_count = get_raw_data_collection(mongo_manager)
        
        if raw_count == 0:
            print("❌ No raw data found in any database")
            print("💡 Please run data collection first")
            mongo_manager.log_pipeline_step('feature_engineering', 'skipped', {
                'reason': 'No raw data available',
                'parent_log_id': log_id
            })
            return
        
        print(f"   ✅ Found {raw_count:,} raw data records")
        
        # Step 2: Run YOUR feature engineering script
        print("\n2️⃣ RUNNING FEATURE ENGINEERING SCRIPT")
        print("-" * 40)
        
        result = run_your_feature_engineering()
        
        if not result or not result.get('success', False):
            print("❌ Feature engineering failed")
            mongo_manager.log_pipeline_step('feature_engineering', 'failed', {
                'error': result.get('error', 'Unknown error') if result else 'No result',
                'parent_log_id': log_id
            })
            return
        
        # Step 3: Get statistics from MongoDB
        print("\n3️⃣ COLLECTING FEATURE STATISTICS")
        print("-" * 40)
        
        # Check aqi_features collection in the main database
        main_db = mongo_manager.client[os.getenv("MONGODB_DATABASE", "aqi_predictor")]
        if 'aqi_features' not in main_db.list_collection_names():
            print("⚠️  aqi_features collection not found in main database")
            features_count = 0
        else:
            features_count = main_db['aqi_features'].count_documents({})
            print(f"   💾 Feature records in main DB: {features_count:,}")
            
            # Get sample feature
            sample_feature = main_db['aqi_features'].find_one()
            if sample_feature:
                print(f"   📋 Sample feature columns: {len(sample_feature.keys())}")
                
                # Show feature names (first 10)
                feature_names = list(sample_feature.keys())
                if '_id' in feature_names:
                    feature_names.remove('_id')
                
                print(f"   🔤 Feature names (first 10): {', '.join(feature_names[:10])}")
                if len(feature_names) > 10:
                    print(f"   ... and {len(feature_names) - 10} more features")
        
        # Also check feature store database
        fs_db = mongo_manager.client[mongo_manager.feature_store_db]
        if 'aqi_features' in fs_db.list_collection_names():
            fs_count = fs_db['aqi_features'].count_documents({})
            print(f"   💾 Feature records in feature store: {fs_count:,}")
            features_count = max(features_count, fs_count)
        
        # Step 4: Create version and log
        print("\n4️⃣ CREATING FEATURE VERSION")
        print("-" * 40)
        
        version = datetime.utcnow().strftime("%Y%m%d_%H%M")
        print(f"   🏷️  Feature version: {version}")
        
        # Store version in feature store database
        version_collection = fs_db['feature_versions']
        version_collection.insert_one({
            'version': version,
            'timestamp': datetime.utcnow(),
            'feature_count': result.get('feature_count', 0),
            'record_count': result.get('record_count', 0),
            'source_script': 'features.py',
            'raw_data_records': raw_count
        })
        
        print(f"   📅 Version stored: {version}")
        
        # Step 5: Log success
        print("\n5️⃣ LOGGING EXECUTION")
        print("-" * 40)
        
        mongo_manager.log_pipeline_step('feature_engineering', 'completed', {
            'success': True,
            'feature_version': version,
            'feature_count': result.get('feature_count', 0),
            'record_count': result.get('record_count', 0),
            'raw_data_records': raw_count,
            'output_lines': result.get('output_lines', 0),
            'parent_log_id': log_id,
            'city': 'Karachi'
        })
        
        print("\n" + "=" * 60)
        print("✅ FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Feature engineering pipeline failed: {str(e)}")
        traceback.print_exc()
        
        mongo_manager.log_pipeline_step('feature_engineering', 'failed', {
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