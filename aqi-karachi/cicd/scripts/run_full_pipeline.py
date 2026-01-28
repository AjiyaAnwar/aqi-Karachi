#!/usr/bin/env python
"""
Full Pipeline Runner
Runs complete end-to-end pipeline
"""
import os
import sys
import time
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

# ==================== CONFIGURATION ====================
CITY_NAME = os.getenv("CITY_NAME", "Karachi")

# ==================== HELPER FUNCTIONS ====================
def run_step(step_name, script_name, mongo_manager, parent_log_id=None):
    """Run a pipeline step with proper logging"""
    print(f"\n{'='*60}")
    print(f"🚀 STEP: {step_name.replace('_', ' ').title()}")
    print(f"{'='*60}")
    
    # Log start
    step_log_id = mongo_manager.log_pipeline_step(step_name, 'started', {
        'parent_log_id': parent_log_id,
        'step_name': step_name
    })
    
    start_time = time.time()
    
    try:
        # Import and run the module
        sys.path.append(str(project_root / 'cicd' / 'scripts'))
        
        try:
            # Dynamic import
            module_name = script_name.replace('.py', '')
            module = __import__(module_name)
            
            # Reload to ensure fresh import
            import importlib
            module = importlib.reload(module)
            
        except ImportError:
            print(f"❌ Failed to import module: {script_name}")
            raise
        
        # Run the step
        result = module.main()  # Assuming each script has a main() function
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log success
        mongo_manager.log_pipeline_step(step_name, 'completed', {
            'duration_seconds': round(duration, 2),
            'result': result if result is not None else 'success',
            'parent_log_id': parent_log_id,
            'step_log_id': step_log_id,
            'step_name': step_name
        })
        
        print(f"✅ {step_name.replace('_', ' ').title()} completed in {duration:.1f} seconds")
        return True, duration, step_log_id
        
    except Exception as e:
        # Calculate duration
        duration = time.time() - start_time
        
        print(f"❌ {step_name.replace('_', ' ').title()} failed: {str(e)}")
        import traceback
        error_details = traceback.format_exc()
        
        # Log failure
        mongo_manager.log_pipeline_step(step_name, 'failed', {
            'duration_seconds': round(duration, 2),
            'error': str(e),
            'traceback': error_details,
            'parent_log_id': parent_log_id,
            'step_log_id': step_log_id,
            'step_name': step_name
        })
        
        return False, duration, step_log_id

def check_prerequisites(mongo_manager):
    """Check if all prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    prerequisites = {
        'mongodb_connection': False,
        'environment_variables': False,
        'required_directories': False
    }
    
    # Check MongoDB connection
    try:
        mongo_manager.client.admin.command('ping')
        prerequisites['mongodb_connection'] = True
        print("   ✅ MongoDB connection: OK")
    except Exception as e:
        print(f"   ❌ MongoDB connection: Failed - {str(e)}")
    
    # Check environment variables
    required_env_vars = ['MONGODB_URI', 'CITY_NAME']
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"   ❌ Missing environment variables: {', '.join(missing_vars)}")
    else:
        prerequisites['environment_variables'] = True
        print("   ✅ Environment variables: OK")
    
    # Check required directories
    required_dirs = ['cicd/scripts', 'eda_output', 'logs']
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"   ⚠️  Missing directories: {', '.join(missing_dirs)}")
        # Try to create them
        for dir_path in missing_dirs:
            full_path = project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"   📁 Created directory: {dir_path}")
        prerequisites['required_directories'] = True
    else:
        prerequisites['required_directories'] = True
        print("   ✅ Required directories: OK")
    
    return all(prerequisites.values()), prerequisites

# ==================== MAIN FUNCTION ====================
def main():
    """Main full pipeline function"""
    print("=" * 80)
    print("🚀 FULL ML PIPELINE EXECUTION")
    print(f"📍 City: {CITY_NAME}")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Get MongoDB URI from environment
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        print("❌ MONGODB_URI environment variable not set")
        sys.exit(1)
    
    # Initialize MongoDB manager
    try:
        mongo_manager = MongoDBManager(mongodb_uri)
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {str(e)}")
        sys.exit(1)
    
    # Check prerequisites
    print("\n🔍 PREREQUISITES CHECK")
    print("-" * 40)
    prerequisites_ok, prerequisites = check_prerequisites(mongo_manager)
    
    if not prerequisites_ok:
        print("❌ Prerequisites not met. Exiting.")
        sys.exit(1)
    
    print("✅ All prerequisites met")
    
    # Log pipeline start
    pipeline_log_id = mongo_manager.log_pipeline_step('full_pipeline', 'started', {
        'city': CITY_NAME,
        'run_id': os.getenv('GITHUB_RUN_ID', 'local'),
        'commit_sha': os.getenv('GITHUB_SHA', 'local'),
        'trigger': os.getenv('GITHUB_EVENT_NAME', 'manual'),
        'environment': 'github_actions' if os.getenv('GITHUB_ACTIONS') == 'true' else 'local',
        'prerequisites': prerequisites
    })
    
    # Define pipeline steps in order
    pipeline_steps = [
        ('data_collection', 'run_data_collection'),
        ('feature_engineering', 'run_feature_engineering'),
        ('eda', 'run_eda'),
        ('model_training', 'run_model_training')
    ]
    
    print(f"\n📋 PIPELINE STEPS ({len(pipeline_steps)} total):")
    for i, (step_name, script_name) in enumerate(pipeline_steps, 1):
        print(f"  {i:2d}. {step_name.replace('_', ' ').title():25} ({script_name}.py)")
    
    print("\n" + "=" * 80)
    print("🏃 STARTING PIPELINE EXECUTION")
    print("=" * 80)
    
    # Execute pipeline steps
    results = {}
    total_duration = 0
    successful_steps = 0
    step_logs = {}
    
    for step_name, script_name in pipeline_steps:
        success, duration, step_log_id = run_step(
            step_name, 
            script_name, 
            mongo_manager, 
            parent_log_id=pipeline_log_id
        )
        
        results[step_name] = {
            'success': success,
            'duration': duration,
            'log_id': step_log_id
        }
        step_logs[step_name] = step_log_id
        total_duration += duration
        
        if success:
            successful_steps += 1
        else:
            print(f"\n⚠️  Pipeline failed at step: {step_name}")
            
            # Decision: Continue or stop?
            # For now, we'll continue but mark as failed
            continue
        
        # Brief pause between steps
        if step_name != pipeline_steps[-1][0]:  # Not the last step
            print(f"\n⏱️  Waiting 5 seconds before next step...")
            time.sleep(5)
    
    # Pipeline completion
    print("\n" + "=" * 80)
    print("📊 PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    
    # Print step results
    print(f"\n📈 STEP RESULTS:")
    print("-" * 40)
    
    for step_name, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        duration = result['duration']
        step_name_display = step_name.replace('_', ' ').title()
        print(f"{step_name_display:25} {status:15} {duration:6.1f}s")
    
    # Print overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print("-" * 40)
    
    print(f"⏱️  Total Duration:     {total_duration:.1f} seconds")
    print(f"📈 Steps Completed:     {successful_steps}/{len(pipeline_steps)}")
    print(f"✅ Success Rate:        {(successful_steps/len(pipeline_steps)*100):.1f}%")
    
    if successful_steps == len(pipeline_steps):
        print(f"🎯 Status:              COMPLETE SUCCESS")
    elif successful_steps > 0:
        print(f"⚠️  Status:              PARTIAL SUCCESS")
    else:
        print(f"❌ Status:              COMPLETE FAILURE")
    
    # Get pipeline statistics
    print(f"\n📊 DATABASE STATISTICS:")
    print("-" * 40)
    
    stats = mongo_manager.get_data_statistics()
    
    print(f"📈 Raw Data Records:    {stats['raw_data_count']:,}")
    print(f"🏷️  Feature Versions:    {stats['feature_versions_count']:,}")
    print(f"🤖 Models in Registry:  {stats['models_count']:,}")
    print(f"🚀 Production Models:   {len(stats['production_models']):,}")
    print(f"📝 Pipeline Logs:       {stats['pipeline_logs_count']:,}")
    
    if stats['latest_feature_version']:
        print(f"🔄 Latest Feature Version: {stats['latest_feature_version']['version']}")
    
    # Show production models
    if stats['production_models']:
        print(f"\n🏆 PRODUCTION MODELS:")
        print("-" * 40)
        for prod_model in stats['production_models']:
            model_name = prod_model['model_name']
            feature_version = prod_model['feature_version']
            r2 = prod_model['metrics'].get('r2', prod_model['metrics'].get('score', 0))
            timestamp = prod_model['timestamp'].strftime('%Y-%m-%d')
            print(f"  • {model_name:20} v{feature_version:15} R²: {r2:.4f} ({timestamp})")
    
    # Log pipeline completion
    print(f"\n📝 LOGGING RESULTS:")
    print("-" * 40)
    
    pipeline_status = 'completed' if successful_steps == len(pipeline_steps) else 'completed_with_errors'
    
    final_log_id = mongo_manager.log_pipeline_step('full_pipeline', pipeline_status, {
        'total_duration': round(total_duration, 2),
        'successful_steps': successful_steps,
        'total_steps': len(pipeline_steps),
        'success_rate': (successful_steps/len(pipeline_steps)*100),
        'step_results': {k: v['success'] for k, v in results.items()},
        'step_durations': {k: v['duration'] for k, v in results.items()},
        'step_logs': step_logs,
        'pipeline_log_id': pipeline_log_id,
        'database_stats': {
            'raw_data_count': stats['raw_data_count'],
            'feature_versions_count': stats['feature_versions_count'],
            'models_count': stats['models_count'],
            'production_models_count': len(stats['production_models'])
        },
        'city': CITY_NAME
    })
    
    print(f"✅ Results logged to MongoDB")
    print(f"📋 Pipeline log ID: {pipeline_log_id}")
    print(f"📋 Final log ID: {final_log_id}")
    
    # Final message
    print("\n" + "=" * 80)
    if successful_steps == len(pipeline_steps):
        print("🎉 FULL PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    elif successful_steps > 0:
        print("⚠️  PIPELINE COMPLETED WITH SOME ERRORS")
    else:
        print("❌ PIPELINE FAILED COMPLETELY")
    
    print(f"🕒 Started:  {datetime.fromtimestamp(time.time() - total_duration).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🕒 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print("=" * 80)
    
    # Exit with appropriate code
    if successful_steps == len(pipeline_steps):
        print("🚀 Exiting with success code (0)")
        sys.exit(0)
    elif successful_steps > 0:
        print("⚠️  Exiting with warning code (1)")
        sys.exit(1)
    else:
        print("❌ Exiting with error code (2)")
        sys.exit(2)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    # Print environment info
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Project Root: {project_root}")
    print(f"🌍 Environment: {'GitHub Actions' if os.getenv('GITHUB_ACTIONS') == 'true' else 'Local'}")
    print(f"📍 City: {CITY_NAME}")
    
    # Run main function
    main()