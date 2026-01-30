#!/usr/bin/env python
"""
FIXED Model Training Runner for CI/CD
Runs the FIXED runallmodels.py orchestrator
"""
import os
import sys
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
import json
import re

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

def run_orchestrator():
    """Run the FIXED runallmodels.py orchestrator"""
    print("🤖 Running 3h Recursive Model Orchestrator...")
    
    orchestrator_path = project_root / 'model_training' / 'runallmodels.py'
    
    if not orchestrator_path.exists():
        print(f"❌ Orchestrator not found: {orchestrator_path}")
        return None
    
    try:
        print("   🎯 Strategy: 3h direct → 72h recursive")
        print("   📊 Expected R²: >0.6 (positive)")
        
        # Run orchestrator - it will handle user input internally
        process = subprocess.run(
            [sys.executable, str(orchestrator_path)],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=600,  # 10 minute timeout
            input="1\n"   # Always use choice 1 (3h Recursive)
        )
        
        print("📊 Orchestrator output:")
        print("-" * 40)
        
        success = False
        r2_score = None
        forecasts_generated = 0
        
        if process.stdout:
            lines = process.stdout.strip().split('\n')
            
            # Show important lines
            for line in lines:
                if any(keyword in line for keyword in 
                       ['R²:', 'Test R²:', 'SUCCESS', 'forecasts', 'Ensemble', '3h Recursive', 'COMPLETED']):
                    print(f"  {line}")
                
                # Extract R²
                if 'R²:' in line or 'Test R²:' in line or 'R² Score:' in line:
                    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                    if numbers:
                        r2_score = float(numbers[0])
                
                # Count forecasts
                if 'forecasts' in line and ('Saved' in line or 'Created' in line or 'forecasts:' in line):
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        forecasts_generated = max(forecasts_generated, int(numbers[0]))
            
            print(f"  ... processed {len(lines)} lines total")
        
        print("-" * 40)
        
        if process.returncode == 0:
            print("✅ Orchestrator completed successfully")
            
            # Check for success indicators
            success_indicators = [
                '3H RECURSIVE PIPELINE COMPLETED',
                'PIPELINE COMPLETED SUCCESSFULLY',
                'forecasts saved',
                'R²: 0.'
            ]
            
            stdout_text = process.stdout.lower() if process.stdout else ""
            success = any(indicator.lower() in stdout_text for indicator in success_indicators)
            
            if r2_score is not None:
                if r2_score > 0:
                    print(f"🎉 Model has positive R²: {r2_score:.4f}")
                else:
                    print(f"⚠️  Model has negative R²: {r2_score:.4f}")
                    success = False
            
            result = {
                'success': success,
                'r2_score': r2_score,
                'forecasts_generated': forecasts_generated,
                'output_lines': len(lines) if process.stdout else 0,
                'timestamp': datetime.now().isoformat(),
                'strategy': '3h_recursive_72h'
            }
            
            return result
        else:
            print(f"❌ Orchestrator failed with code {process.returncode}")
            if process.stderr:
                print("Error output:")
                for line in process.stderr.strip().split('\n')[-10:]:
                    if line.strip():
                        print(f"  ❌ {line.strip()}")
            
            return {
                'success': False,
                'error_code': process.returncode,
                'error_message': process.stderr[:200] if process.stderr else "Unknown error",
                'strategy': '3h_recursive_72h'
            }
            
    except subprocess.TimeoutExpired:
        print("❌ Orchestrator timed out after 10 minutes")
        return {
            'success': False,
            'error': 'Timeout',
            'strategy': '3h_recursive_72h'
        }
    except Exception as e:
        print(f"❌ Error running orchestrator: {str(e)}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'strategy': '3h_recursive_72h'
        }

def check_forecasts(mongo_manager):
    """Check forecasts generated by GOOD model"""
    print("\n🔍 Checking Forecast Results...")
    
    main_db = mongo_manager.client[mongo_manager.db_name]
    
    # Collections from GOOD model
    good_collections = [
        'ml_recursive_forecasts',  # From train_models.py
        'ml_forecasts_3day',
        'ensemble_forecasts_3day',
        'simple_forecasts',
        'timeseries_forecasts_3day'
    ]
    
    total_forecasts = 0
    collection_counts = {}
    
    for coll_name in good_collections:
        if coll_name in main_db.list_collection_names():
            count = main_db[coll_name].count_documents({})
            if count > 0:
                print(f"   📊 {coll_name}: {count} forecasts")
                total_forecasts += count
                collection_counts[coll_name] = count
    
    print(f"   📈 Total GOOD forecasts: {total_forecasts}")
    
    # Show sample forecasts
    if 'ml_recursive_forecasts' in main_db.list_collection_names():
        sample = list(main_db['ml_recursive_forecasts'].find(
            {}, 
            {'_id': 0, 'predicted_aqi': 1, 'date': 1, 'hours_ahead': 1, 'category': 1}
        ).limit(3))
        
        if sample:
            print(f"\n   🎯 Sample 3h Recursive Forecasts:")
            for forecast in sample:
                print(f"      {forecast.get('hours_ahead', 'N/A')}h: {forecast.get('predicted_aqi', 'N/A'):.1f} AQI ({forecast.get('category', 'N/A')})")
    
    return total_forecasts, collection_counts

def check_model_registry(mongo_manager):
    """Check model registry for GOOD models"""
    print("\n🔍 Checking Model Registry for GOOD models...")
    
    mr_db = mongo_manager.client[mongo_manager.model_registry_db]
    
    total_good_models = 0
    good_models_info = []
    
    # Check multiple collections
    collections_to_check = ['model_registry', 'models', 'models_3h']
    
    for collection in collections_to_check:
        if collection in mr_db.list_collection_names():
            # Find models with positive R²
            query = {
                '$or': [
                    {'metrics.r2_score': {'$gt': 0}},
                    {'metrics.Test R²': {'$gt': 0}},
                    {'test_r2': {'$gt': 0}},
                    {'metrics.Test_R2': {'$gt': 0}}
                ]
            }
            
            good_models = list(mr_db[collection].find(query).sort('created_at', -1).limit(5))
            
            for model in good_models:
                total_good_models += 1
                
                # Extract R²
                r2 = None
                if 'metrics' in model:
                    if 'r2_score' in model['metrics']:
                        r2 = model['metrics']['r2_score']
                    elif 'Test R²' in model['metrics']:
                        r2 = model['metrics']['Test R²']
                    elif 'Test_R2' in model['metrics']:
                        r2 = model['metrics']['Test_R2']
                elif 'test_r2' in model:
                    r2 = model['test_r2']
                
                model_info = {
                    'name': model.get('model_name', 'Unknown'),
                    'r2': r2,
                    'created': model.get('created_at', 'N/A'),
                    'type': model.get('model_type', 'N/A'),
                    'collection': collection,
                    'id': str(model.get('_id', 'N/A'))
                }
                good_models_info.append(model_info)
    
    print(f"   📊 GOOD models found: {total_good_models}")
    
    if good_models_info:
        print(f"\n   🤖 Latest GOOD Models:")
        for i, model in enumerate(good_models_info[:3], 1):
            r2_str = f"{model['r2']:.4f}" if model['r2'] is not None else "N/A"
            print(f"   {i}. {model['name']}")
            print(f"      R²: {r2_str}")
            print(f"      Type: {model['type']}")
            print(f"      Created: {model['created']}")
            print(f"      ID: {model['id'][:10]}...")
    
    return total_good_models, good_models_info

def check_feature_store(mongo_manager):
    """Check feature store for GOOD features"""
    print("\n🔍 Checking Feature Store...")
    
    fs_db = mongo_manager.client[mongo_manager.feature_store_db]
    
    # Check for simple features (used by GOOD model)
    feature_collections = ['aqi_features_simple', 'aqi_features']
    features_found = False
    feature_count = 0
    feature_collection = None
    
    for coll_name in feature_collections:
        if coll_name in fs_db.list_collection_names():
            count = fs_db[coll_name].count_documents({})
            if count > 0:
                feature_count = count
                feature_collection = coll_name
                features_found = True
                break
    
    if not features_found:
        print("❌ No features found in Feature Store")
        return 0, None, False
    
    print(f"✅ Found {feature_count:,} feature records in {feature_collection}")
    
    # Check for target_3h (required for GOOD model)
    sample_feature = fs_db[feature_collection].find_one()
    has_target_3h = False
    if sample_feature:
        targets = [col for col in sample_feature.keys() if 'target' in col]
        if 'target_3h' in targets:
            print(f"🎯 Found target_3h column (required for 3h model)")
            has_target_3h = True
        elif targets:
            print(f"🎯 Target columns: {', '.join(targets)}")
        else:
            print("⚠️  No target columns found")
    
    return feature_count, feature_collection, has_target_3h

def main():
    """Main model training function for CI/CD - FIXED version"""
    print("=" * 70)
    print("🤖 CI/CD 3H RECURSIVE MODEL TRAINING PIPELINE")
    print("📍 City: Karachi")
    print("🎯 Strategy: 3h direct → 72h recursive")
    print("🕒 Time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    # Get MongoDB URI from environment
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        print("❌ MONGODB_URI environment variable not set")
        sys.exit(1)
    
    # Initialize MongoDB manager
    try:
        mongo_manager = MongoDBManager(mongodb_uri)
        print(f"✅ Connected to MongoDB")
        print(f"   Main DB: {mongo_manager.db_name}")
        print(f"   Feature Store: {mongo_manager.feature_store_db}")
        print(f"   Model Registry: {mongo_manager.model_registry_db}")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {str(e)}")
        sys.exit(1)
    
    # Log start
    log_id = mongo_manager.log_pipeline_step('model_training', 'started', {
        'city': 'Karachi',
        'run_type': 'github_actions' if os.getenv('GITHUB_ACTIONS') == 'true' else 'local',
        'script': 'runallmodels.py',
        'strategy': '3h_recursive_72h',
        'timestamp': datetime.now().isoformat()
    })
    
    try:
        # Step 1: Check feature store
        print("\n1️⃣ CHECKING FEATURE STORE")
        print("-" * 40)
        
        feature_count, feature_collection, has_target_3h = check_feature_store(mongo_manager)
        
        if feature_count == 0:
            print("❌ No features found in Feature Store")
            print("💡 Please run feature engineering first")
            mongo_manager.log_pipeline_step('model_training', 'skipped', {
                'reason': 'No features in Feature Store',
                'parent_log_id': log_id
            })
            return
        
        if not has_target_3h:
            print("⚠️  Warning: target_3h not found. 3h model may not work optimally.")
        
        # Step 2: Run FIXED orchestrator
        print("\n2️⃣ RUNNING 3H RECURSIVE MODEL ORCHESTRATOR")
        print("-" * 40)
        
        result = run_orchestrator()
        
        if not result or not result.get('success', False):
            print("❌ 3h Recursive model training failed")
            mongo_manager.log_pipeline_step('model_training', 'failed', {
                'error': result.get('error', 'Unknown error') if result else 'No result',
                'r2_score': result.get('r2_score') if result else None,
                'parent_log_id': log_id
            })
            return
        
        print(f"✅ 3h Recursive model trained successfully")
        if result.get('r2_score'):
            print(f"🎯 Model R²: {result['r2_score']:.4f}")
        
        # Step 3: Check GOOD forecasts
        print("\n3️⃣ CHECKING FORECAST RESULTS")
        print("-" * 40)
        
        total_forecasts, collection_counts = check_forecasts(mongo_manager)
        
        if total_forecasts == 0:
            print("⚠️  No forecasts generated by GOOD model")
        else:
            print(f"✅ {total_forecasts} forecasts ready for dashboard")
        
        # Step 4: Check Model Registry for GOOD models
        print("\n4️⃣ CHECKING MODEL REGISTRY")
        print("-" * 40)
        
        model_count, models_info = check_model_registry(mongo_manager)
        
        # Step 5: Log success
        print("\n5️⃣ LOGGING EXECUTION")
        print("-" * 40)
        
        final_result = {
            'success': True,
            'feature_count': feature_count,
            'feature_collection': feature_collection,
            'has_target_3h': has_target_3h,
            'forecasts_generated': total_forecasts,
            'forecast_collections': collection_counts,
            'models_in_registry': model_count,
            'model_r2': result.get('r2_score'),
            'orchestrator_result': result,
            'good_models': models_info[:3] if models_info else [],
            'parent_log_id': log_id,
            'city': 'Karachi',
            'strategy': '3h_recursive_72h',
            'duration_seconds': (datetime.now() - datetime.fromisoformat(result.get('timestamp', datetime.now().isoformat()))).total_seconds()
        }
        
        mongo_manager.log_pipeline_step('model_training', 'completed', final_result)
        
        print("\n" + "=" * 70)
        print("✅ 3H RECURSIVE CI/CD PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print(f"\n📊 SUMMARY:")
        print(f"  Features: {feature_count:,} from {feature_collection}")
        print(f"  Model R²: {result.get('r2_score', 'N/A'):.4f}")
        print(f"  Forecasts: {total_forecasts}")
        print(f"  GOOD Models in Registry: {model_count}")
        
        print(f"\n🎯 STRATEGY USED:")
        print(f"  3h direct prediction → 72h recursive forecast")
        print(f"  Ensemble: 70% 3h Model + 30% Time Series")
        
        print(f"\n📁 Collections with forecasts:")
        for coll_name, count in collection_counts.items():
            print(f"  • {coll_name}: {count} forecasts")
        
        # Save summary to file for GitHub Actions
        if os.getenv('GITHUB_ACTIONS') == 'true':
            summary_file = project_root / 'cicd' / 'training_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(final_result, f, indent=2, default=str)
            
            print(f"\n📄 Summary saved to: {summary_file}")
            
            # Output result for GitHub Actions
            print(f"::set-output name=r2_score::{result.get('r2_score', 0)}")
            print(f"::set-output name=forecasts::{total_forecasts}")
            print(f"::set-output name=models::{model_count}")
            print(f"::set-output name=success::true")
        
    except Exception as e:
        print(f"\n❌ CI/CD pipeline failed: {str(e)}")
        traceback.print_exc()
        
        mongo_manager.log_pipeline_step('model_training', 'failed', {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'parent_log_id': log_id,
            'strategy': '3h_recursive_72h'
        })
        
        print("\n" + "=" * 60)
        print("❌ 3H RECURSIVE PIPELINE FAILED")
        print("=" * 60)
        
        if os.getenv('GITHUB_ACTIONS') == 'true':
            print(f"::set-output name=success::false")
            print(f"::set-output name=error::{str(e)[:100]}")
        
        sys.exit(1)

if __name__ == "__main__":
    # Print environment info
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Project Root: {project_root}")
    print(f"🌍 Environment: {'GitHub Actions' if os.getenv('GITHUB_ACTIONS') == 'true' else 'Local'}")
    print(f"🕒 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run main function
    main()