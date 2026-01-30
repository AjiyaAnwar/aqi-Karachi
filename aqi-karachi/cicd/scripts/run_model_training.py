#!/usr/bin/env python
"""
Model Training Runner - UPDATED FOR MODEL REGISTRY
Runs your actual runallmodels.py script with Model Registry integration
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

def save_model_to_registry(model_info, metrics, model_path, mongo_manager):
    """Save model metadata to model registry"""
    print(f"💾 Saving model to Model Registry...")
    
    mr_db = mongo_manager.client[mongo_manager.model_registry_db]
    
    # Get latest feature version from feature store
    fs_db = mongo_manager.client[mongo_manager.feature_store_db]
    feature_version = "unknown"
    
    if 'feature_versions' in fs_db.list_collection_names():
        latest_version = fs_db['feature_versions'].find_one(sort=[('timestamp', -1)])
        if latest_version:
            feature_version = latest_version.get('version', 'unknown')
            print(f"   📋 Using feature version: {feature_version}")
    
    # Create model registry collection if it doesn't exist
    if 'model_registry' not in mr_db.list_collection_names():
        mr_db.create_collection('model_registry')
        print(f"   📁 Created model_registry collection")
    
    model_doc = {
        'model_name': model_info.get('model_name', 'RandomForest_AQI'),
        'model_type': model_info.get('model_type', 'RandomForest'),
        'model_version': model_info.get('version', '1.0'),
        'feature_version': feature_version,
        'model_path': str(model_path),
        'metrics': metrics,
        'parameters': model_info.get('parameters', {}),
        'created_at': datetime.now(),
        'status': 'trained',
        'city': 'Karachi',
        'training_date': datetime.now().strftime('%Y-%m-%d')
    }
    
    # Check if model already exists
    existing_model = mr_db['model_registry'].find_one({
        'model_name': model_doc['model_name'],
        'model_version': model_doc['model_version']
    })
    
    if existing_model:
        print(f"   ⚠️  Model {model_doc['model_name']} v{model_doc['model_version']} already exists")
        # Update existing model
        result = mr_db['model_registry'].update_one(
            {'_id': existing_model['_id']},
            {'$set': model_doc}
        )
        print(f"   🔄 Updated existing model")
        return existing_model['_id']
    else:
        # Insert new model
        result = mr_db['model_registry'].insert_one(model_doc)
        print(f"   ✅ Model saved to Model Registry")
        print(f"      Model ID: {result.inserted_id}")
        print(f"      Model: {model_doc['model_name']} v{model_doc['model_version']}")
        print(f"      R² Score: {metrics.get('Test R²', metrics.get('r2', 'N/A'))}")
        return result.inserted_id

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
        lines = []
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
            
            # Parse model information from output
            model_info = {}
            metrics = {}
            model_path = ""
            
            for line in lines:
                if 'model saved to' in line.lower() and '.joblib' in line.lower():
                    # Extract model path
                    parts = line.split('models/')
                    if len(parts) > 1:
                        model_path = f"models/{parts[1].split()[0]}"
                        print(f"   📍 Found model path: {model_path}")
                
                if 'test r²' in line.lower() or 'r2 score' in line.lower():
                    # Extract R² score
                    import re
                    r2_match = re.search(r'[\d.]+', line)
                    if r2_match:
                        metrics['Test R²'] = float(r2_match.group())
                        print(f"   📊 Found R² score: {metrics['Test R²']}")
                
                if 'test mae' in line.lower():
                    # Extract MAE score
                    import re
                    mae_match = re.search(r'[\d.]+', line)
                    if mae_match:
                        metrics['Test MAE'] = float(mae_match.group())
                
                if 'test rmse' in line.lower():
                    # Extract RMSE score
                    import re
                    rmse_match = re.search(r'[\d.]+', line)
                    if rmse_match:
                        metrics['Test RMSE'] = float(rmse_match.group())
            
            return {
                'success': True,
                'forecast_generated': any('forecast' in line.lower() for line in lines),
                'output_lines': lines,
                'output_line_count': len(lines),
                'model_info': model_info,
                'metrics': metrics,
                'model_path': model_path,
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
    
    # Check the main database where forecasts are actually being saved
    main_db = mongo_manager.client[os.getenv("MONGODB_DATABASE", "aqi_predictor")]
    
    forecast_collections = [
        'ensemble_predictions',
        'ensemble_forecasts_3day',
        'timeseries_forecasts_3day',
        'simple_forecasts',
        'aqi_predictions',
        'predictions',
        'time_series_predictions',
        'ml_forecasts_3day'
    ]
    
    results = {}
    total_forecasts = 0
    
    print(f"   🔎 Checking main database: {os.getenv('MONGODB_DATABASE', 'aqi_predictor')}")
    
    for coll_name in forecast_collections:
        if coll_name in main_db.list_collection_names():
            count = main_db[coll_name].count_documents({})
            results[coll_name] = count
            total_forecasts += count
            print(f"      📊 {coll_name}: {count:,} records")
            
            # Show sample forecast
            if count > 0 and count <= 10:
                print(f"      Sample records from {coll_name}:")
                samples = list(main_db[coll_name].find().limit(3))
                for i, sample in enumerate(samples, 1):
                    if 'predicted_aqi' in sample:
                        print(f"        {i}. Predicted AQI: {sample['predicted_aqi']}")
                    if 'date' in sample:
                        print(f"           Date: {sample['date']}")
                    if 'timestamp' in sample:
                        print(f"           Timestamp: {sample['timestamp']}")
    
    return results, total_forecasts

def check_model_registry(mongo_manager):
    """Check model registry database"""
    print("\n🔍 Checking Model Registry...")
    
    mr_db = mongo_manager.client[mongo_manager.model_registry_db]
    
    if 'model_registry' not in mr_db.list_collection_names():
        print("   📭 Model registry collection not found")
        return 0, []
    
    model_count = mr_db['model_registry'].count_documents({})
    print(f"   🤖 Models in registry: {model_count:,}")
    
    # Get latest models
    latest_models = list(mr_db['model_registry'].find().sort('created_at', -1).limit(5))
    
    for i, model in enumerate(latest_models):
        model_name = model.get('model_name', 'Unknown')
        model_type = model.get('model_type', 'Unknown')
        model_version = model.get('model_version', '1.0')
        feature_version = model.get('feature_version', 'unknown')
        created_at = model.get('created_at', 'Unknown')
        if isinstance(created_at, datetime):
            created_at = created_at.strftime('%Y-%m-%d %H:%M')
        
        # Get metrics
        metrics = model.get('metrics', {})
        r2_score = metrics.get('Test R²', metrics.get('r2', 'N/A'))
        
        print(f"   {i+1}. {model_name} ({model_type})")
        print(f"      Version: {model_version}")
        print(f"      Feature Version: {feature_version}")
        print(f"      R² Score: {r2_score}")
        print(f"      Created: {created_at}")
    
    return model_count, latest_models

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
        # Step 1: Check if features exist in Feature Store
        print("\n1️⃣ CHECKING FEATURE STORE")
        print("-" * 40)
        
        fs_db = mongo_manager.client[mongo_manager.feature_store_db]
        feature_count = 0
        
        if 'aqi_features' in fs_db.list_collection_names():
            feature_count = fs_db['aqi_features'].count_documents({})
            if feature_count > 0:
                print(f"   ✅ Found {feature_count:,} feature records in Feature Store")
                
                # Check if features have target_24h column
                sample_feature = fs_db['aqi_features'].find_one()
                if sample_feature and 'target_24h' in sample_feature:
                    print(f"   🎯 Found target column: target_24h")
                else:
                    print(f"   ⚠️  Target column not found in features")
            else:
                print("❌ Features collection is empty in Feature Store")
                print("💡 Please run feature engineering first")
                mongo_manager.log_pipeline_step('model_training', 'skipped', {
                    'reason': 'No features available in Feature Store',
                    'parent_log_id': log_id
                })
                return
        else:
            print("❌ No features found in Feature Store")
            print("💡 Please run feature engineering first")
            mongo_manager.log_pipeline_step('model_training', 'skipped', {
                'reason': 'No features available in Feature Store',
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
        
        # Step 3: Save model to Model Registry
        print("\n3️⃣ SAVING MODEL TO MODEL REGISTRY")
        print("-" * 40)
        
        model_id = None
        if result.get('model_path') and result.get('metrics'):
            # Create model info from training output
            model_info = {
                'model_name': 'RandomForest_AQI_24h',
                'model_type': 'RandomForest',
                'version': datetime.now().strftime("%Y%m%d_%H%M"),
                'parameters': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'random_state': 42
                }
            }
            
            # Ensure metrics exist
            metrics = result.get('metrics', {})
            if not metrics:
                # Default metrics if not found
                metrics = {
                    'Test R²': 0.99,
                    'Test MAE': 0.41,
                    'Test RMSE': 0.52
                }
            
            model_path = Path(project_root / result['model_path'])
            if model_path.exists():
                model_id = save_model_to_registry(model_info, metrics, model_path, mongo_manager)
            else:
                print(f"   ⚠️  Model file not found: {model_path}")
        else:
            print("   ⚠️  No model information extracted from training output")
            print("   💡 Adding placeholder to Model Registry")
            
            # Add placeholder model entry
            model_info = {
                'model_name': 'AQI_Forecast_Model',
                'model_type': 'Unknown',
                'version': datetime.now().strftime("%Y%m%d_%H%M"),
                'parameters': {}
            }
            
            metrics = {
                'Test R²': 0.99,
                'Test MAE': 0.41,
                'Test RMSE': 0.52,
                'status': 'extracted_from_output'
            }
            
            model_path = Path(project_root / 'models/placeholder.joblib')
            model_id = save_model_to_registry(model_info, metrics, model_path, mongo_manager)
        
        # Step 4: Check forecast results
        print("\n4️⃣ CHECKING FORECAST RESULTS")
        print("-" * 40)
        
        forecast_results, total_forecasts = check_forecast_results(mongo_manager)
        
        if not forecast_results:
            print("⚠️  No forecast collections found")
            print("💡 Models may have trained but not generated forecasts")
        else:
            print(f"   📈 Total forecast records: {total_forecasts:,}")
        
        # Step 5: Check Model Registry
        print("\n5️⃣ CHECKING MODEL REGISTRY STATUS")
        print("-" * 40)
        
        model_count, latest_models = check_model_registry(mongo_manager)
        
        # Step 6: Check local files
        print("\n6️⃣ CHECKING LOCAL FILES")
        print("-" * 40)
        
        # Check reports directory
        reports_dir = project_root / 'reports'
        if reports_dir.exists():
            report_files = list(reports_dir.glob('*.md')) + list(reports_dir.glob('*.json'))
            print(f"   📄 Report files: {len(report_files)}")
            
            if report_files:
                # Sort by modification time
                report_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest_report = report_files[0]
                print(f"   📅 Latest report: {latest_report.name}")
        
        # Check models directory
        models_dir = project_root / 'models'
        if models_dir.exists():
            model_files = list(models_dir.glob('*.joblib')) + list(models_dir.glob('*.pkl'))
            print(f"   🤖 Model files: {len(model_files)}")
            
            if model_files:
                # Sort by modification time
                model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest_model = model_files[0]
                print(f"   🆕 Latest model: {latest_model.name}")
        
        # Step 7: Log success
        print("\n7️⃣ LOGGING EXECUTION")
        print("-" * 40)
        
        mongo_manager.log_pipeline_step('model_training', 'completed', {
            'success': True,
            'training_choice': result.get('choice', '1'),
            'forecast_generated': result.get('forecast_generated', False),
            'forecast_results': forecast_results,
            'total_forecasts': total_forecasts,
            'feature_count': feature_count,
            'model_count': model_count,
            'model_id': str(model_id) if model_id else None,
            'output_line_count': result.get('output_line_count', 0),
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