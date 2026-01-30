#!/usr/bin/env python
"""
Model Training Runner - INTEGRATED WITH YOUR CODE
Runs your actual runallmodels.py script
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

def run_your_model_training(choice='1'):
    """Run your actual model training script with specific choice"""
    print("🤖 Running your model training script...")
    
    script_path = project_root / 'model_training' / 'runallmodels.py'
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return None
    
    try:
        # Run your script using subprocess with input
        print(f"   🎯 Training choice: {choice} (1=Combined, 2=Individual)")
        
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=project_root
        )
        
        # Send the choice (1 for Combined Training, 2 for Individual)
        stdout, stderr = process.communicate(input=f"{choice}\n", timeout=600)  # 10 minute timeout
        
        print("📊 Model training output:")
        print("-" * 40)
        
        # Print relevant output
        if stdout:
            lines = stdout.strip().split('\n')
            
            # Filter and show important lines
            important_lines = []
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in [
                    'training', 'model', 'forecast', 'error', 'warning', 
                    'success', 'failed', 'complete', 'accuracy', 'score',
                    'r2', 'mae', 'rmse', 'result', 'report', 'saved'
                ]):
                    important_lines.append(line.strip())
            
            # Show all important lines (up to 40)
            for line in important_lines[:40]:
                if line:
                    print(f"  {line}")
            
            # Show count of lines processed
            print(f"  ... processed {len(lines)} lines total")
        
        print("-" * 40)
        
        if process.returncode == 0:
            print("✅ Your model training script completed successfully")
            
            # Check for forecast results in output
            forecast_generated = False
            for line in lines:
                if 'forecast' in line.lower() and ('generated' in line.lower() or 'saved' in line.lower()):
                    forecast_generated = True
                    break
            
            return {
                'success': True,
                'forecast_generated': forecast_generated,
                'output_lines': len(lines),
                'choice': choice
            }
        else:
            print(f"❌ Your model training script failed with code {process.returncode}")
            if stderr:
                print("Error output:")
                for line in stderr.strip().split('\n')[-10:]:
                    if line.strip():
                        print(f"  ❌ {line.strip()}")
            
            return {
                'success': False,
                'error_code': process.returncode,
                'error_message': stderr[:200] if stderr else "Unknown error",
                'choice': choice
            }
            
    except subprocess.TimeoutExpired:
        print("❌ Model training timed out after 10 minutes")
        return {
            'success': False,
            'error': 'Timeout',
            'choice': choice
        }
    except Exception as e:
        print(f"❌ Error running model training: {str(e)}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'choice': choice
        }

def check_forecast_results(mongo_manager):
    """Check what forecast collections were created"""
    print("🔍 Checking forecast results...")
    
    db = mongo_manager.client[mongo_manager.model_registry_db]
    collections = db.list_collection_names()
    
    forecast_collections = [
        'ml_forecasts_3day',
        'timeseries_forecasts_3day', 
        'ensemble_forecasts_3day',
        'ml_forecasts',
        'simple_forecasts',
        'aqi_predictions'
    ]
    
    results = {}
    
    for coll_name in forecast_collections:
        if coll_name in collections:
            count = db[coll_name].count_documents({})
            results[coll_name] = count
            print(f"   📊 {coll_name}: {count:,} records")
            
            # Show sample forecast
            if count > 0:
                sample = db[coll_name].find_one()
                if sample:
                    if 'predicted_aqi' in sample:
                        print(f"     Sample AQI: {sample['predicted_aqi']}")
                    if 'date' in sample:
                        print(f"     Sample date: {sample['date']}")
    
    return results

def main():
    """Main model training function - integrates with your code"""
    print("=" * 60)
    print("🤖 MODEL TRAINING PIPELINE")
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
        print(f"   Model Registry: {mongo_manager.model_registry_db}")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {str(e)}")
        sys.exit(1)
    
    # Log start
    log_id = mongo_manager.log_pipeline_step('model_training', 'started', {
        'city': 'Karachi',
        'run_type': 'github_actions' if os.getenv('GITHUB_ACTIONS') == 'true' else 'local',
        'script': 'runallmodels.py'
    })
    
    try:
        # Step 1: Check if features exist
        print("\n1️⃣ CHECKING FOR FEATURES")
        print("-" * 40)
        
        # First check main database
        main_db = mongo_manager.client[os.getenv("MONGODB_DATABASE", "aqi_predictor")]
        feature_count = 0
        features_db = None  # Renamed from fs_db to avoid confusion
        features_collection_name = 'aqi_features'
        
        # Check main database first
        if features_collection_name in main_db.list_collection_names():
            feature_count = main_db[features_collection_name].count_documents({})
            if feature_count > 0:
                print(f"   ✅ Found {feature_count:,} feature records in main database")
                features_db = main_db
        
        # If not found in main database, check feature store
        if feature_count == 0:
            features_db = mongo_manager.client[mongo_manager.feature_store_db]
            if features_collection_name in features_db.list_collection_names():
                feature_count = features_db[features_collection_name].count_documents({})
        
        if feature_count == 0:
            print("❌ No features found in aqi_features collection")
            print("💡 Please run feature engineering first")
            mongo_manager.log_pipeline_step('model_training', 'skipped', {
                'reason': 'No features available',
                'parent_log_id': log_id
            })
            return
        
        print(f"   ✅ Found {feature_count:,} feature records")
        
        # Check if features have target_24h column
        # FIXED: Check if features_db is not None before using it
        if features_db is not None:
            sample_feature = features_db[features_collection_name].find_one()
            if sample_feature and 'target_24h' in sample_feature:
                print(f"   🎯 Found target column: target_24h")
            else:
                print(f"   ⚠️  Target column not found in features")
        else:
            print(f"   ⚠️  Could not access features database")
            mongo_manager.log_pipeline_step('model_training', 'failed', {
                'reason': 'Could not access features database',
                'parent_log_id': log_id
            })
            return
        
        # Step 2: Run YOUR model training script
        print("\n2️⃣ RUNNING MODEL TRAINING SCRIPT")
        print("-" * 40)
        
        # Determine training choice based on schedule
        # For scheduled runs, use Combined Training (choice 1)
        # For manual runs, could be configurable
        training_choice = '1'  # Default to Combined Training
        
        result = run_your_model_training(training_choice)
        
        if not result or not result.get('success', False):
            print("❌ Model training failed")
            
            # Try individual models as fallback
            print("\n🔄 Trying individual models as fallback...")
            fallback_result = run_your_model_training('2')
            
            if not fallback_result or not fallback_result.get('success', False):
                print("❌ Both training methods failed")
                mongo_manager.log_pipeline_step('model_training', 'failed', {
                    'error': result.get('error', 'Unknown error') if result else 'No result',
                    'fallback_error': fallback_result.get('error', 'Unknown error') if fallback_result else 'No fallback result',
                    'parent_log_id': log_id
                })
                return
            else:
                result = fallback_result
        
        # Step 3: Check forecast results
        print("\n3️⃣ CHECKING FORECAST RESULTS")
        print("-" * 40)
        
        forecast_results = check_forecast_results(mongo_manager)
        
        if not forecast_results:
            print("⚠️  No forecast collections found")
            print("💡 Models may have trained but not generated forecasts")
        else:
            total_forecasts = sum(forecast_results.values())
            print(f"   📈 Total forecast records: {total_forecasts:,}")
        
        # Step 4: Check model registry
        print("\n4️⃣ CHECKING MODEL REGISTRY")
        print("-" * 40)
        
        mr_db = mongo_manager.client[mongo_manager.model_registry_db]
        
        if 'model_registry' in mr_db.list_collection_names():
            model_count = mr_db['model_registry'].count_documents({})
            print(f"   🤖 Models in registry: {model_count:,}")
            
            # Get latest models
            latest_models = list(mr_db['model_registry'].find().sort('created_at', -1).limit(5))
            
            for i, model in enumerate(latest_models):
                model_name = model.get('model_name', 'Unknown')
                created_at = model.get('created_at', 'Unknown')
                if isinstance(created_at, datetime):
                    created_at = created_at.strftime('%Y-%m-%d %H:%M')
                r2_score = model.get('metrics', {}).get('Test R²', 'N/A')
                
                print(f"   {i+1}. {model_name}")
                print(f"      Created: {created_at}")
                print(f"      R² Score: {r2_score}")
        else:
            print("   📭 Model registry collection not found")
        
        # Step 5: Check local files
        print("\n5️⃣ CHECKING LOCAL FILES")
        print("-" * 40)
        
        # Check reports directory
        reports_dir = project_root / 'reports'
        if reports_dir.exists():
            report_files = list(reports_dir.glob('*.md'))
            print(f"   📄 Report files: {len(report_files)}")
            
            if report_files:
                latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                print(f"   📅 Latest report: {latest_report.name}")
        
        # Check models directory
        models_dir = project_root / 'models'
        if models_dir.exists():
            model_files = list(models_dir.glob('*.joblib')) + list(models_dir.glob('*.pkl'))
            print(f"   🤖 Model files: {len(model_files)}")
        
        # Step 6: Log success
        print("\n6️⃣ LOGGING EXECUTION")
        print("-" * 40)
        
        mongo_manager.log_pipeline_step('model_training', 'completed', {
            'success': True,
            'training_choice': result.get('choice', '1'),
            'forecast_generated': result.get('forecast_generated', False),
            'forecast_results': forecast_results,
            'feature_count': feature_count,
            'output_lines': result.get('output_lines', 0),
            'parent_log_id': log_id,
            'city': 'Karachi'
        })
        
        print("\n" + "=" * 60)
        print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Model training pipeline failed: {str(e)}")
        traceback.print_exc()
        
        mongo_manager.log_pipeline_step('model_training', 'failed', {
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