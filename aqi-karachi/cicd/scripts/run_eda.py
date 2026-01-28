#!/usr/bin/env python
"""
EDA Runner - INTEGRATED WITH YOUR CODE
Runs your actual eda.py script
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

def run_your_eda():
    """Run your actual EDA script"""
    print("📊 Running your EDA script...")
    
    script_path = project_root / 'notebook' / 'eda.py'
    
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
        
        print("📊 EDA analysis output:")
        print("-" * 40)
        
        # Print relevant output
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            
            # Show summary and key insights
            summary_lines = []
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in [
                    'insight', 'summary', 'analysis', 'found', 'data', 
                    'record', 'feature', 'aqi', 'category', 'distribution',
                    'mean', 'median', 'std', 'range', 'completed', 'saved'
                ]):
                    summary_lines.append(line.strip())
            
            # Show all summary lines (up to 30)
            for line in summary_lines[:30]:
                if line:
                    print(f"  {line}")
            
            # Show count of lines processed
            print(f"  ... processed {len(lines)} lines total")
        
        print("-" * 40)
        
        if result.returncode == 0:
            print("✅ Your EDA script completed successfully")
            
            # Extract insights from output
            insights_count = 0
            for line in lines:
                if 'insight' in line.lower() or 'key finding' in line.lower():
                    insights_count += 1
            
            return {
                'success': True,
                'insights_count': insights_count,
                'output_lines': len(lines)
            }
        else:
            print(f"❌ Your EDA script failed with code {result.returncode}")
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
        print("❌ EDA analysis timed out after 5 minutes")
        return {
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"❌ Error running EDA: {str(e)}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def check_eda_results():
    """Check what EDA results were generated"""
    print("🔍 Checking EDA results...")
    
    results = {
        'plots': [],
        'reports': [],
        'statistics': []
    }
    
    # Check results directory
    results_dir = project_root / 'results' / 'eda_reports'
    if results_dir.exists():
        # Check plots
        plots_dir = results_dir / 'plots'
        if plots_dir.exists():
            plot_files = list(plots_dir.glob('*.png')) + list(plots_dir.glob('*.html'))
            results['plots'] = [f.name for f in plot_files]
            print(f"   🖼️  Plots generated: {len(plot_files)}")
            
            # List first 5 plots
            for plot in plot_files[:5]:
                print(f"     - {plot.name}")
            if len(plot_files) > 5:
                print(f"     ... and {len(plot_files) - 5} more")
        
        # Check reports
        report_files = list(results_dir.glob('*.md')) + list(results_dir.glob('*.txt'))
        results['reports'] = [f.name for f in report_files]
        
        if report_files:
            print(f"   📄 Reports generated: {len(report_files)}")
            for report in report_files:
                print(f"     - {report.name}")
    
    # Check for correlation files
    if results_dir.exists():
        csv_files = list(results_dir.rglob('*.csv'))
        results['statistics'] = [f.name for f in csv_files]
        
        if csv_files:
            print(f"   📊 Statistics files: {len(csv_files)}")
            for csv in csv_files[:3]:
                print(f"     - {csv.name}")
    
    return results

def main():
    """Main EDA function - integrates with your code"""
    print("=" * 60)
    print("📊 EXPLORATORY DATA ANALYSIS")
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
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {str(e)}")
        sys.exit(1)
    
    # Log start
    log_id = mongo_manager.log_pipeline_step('eda', 'started', {
        'city': 'Karachi',
        'run_type': 'github_actions' if os.getenv('GITHUB_ACTIONS') == 'true' else 'local',
        'script': 'eda.py'
    })
    
    try:
        # Step 1: Check if data exists
        print("\n1️⃣ CHECKING FOR DATA")
        print("-" * 40)
        
        fs_db = mongo_manager.client[mongo_manager.feature_store_db]
        
        # Check aqi_features collection (preferred)
        if 'aqi_features' in fs_db.list_collection_names():
            feature_count = fs_db['aqi_features'].count_documents({})
            print(f"   ✅ Found {feature_count:,} feature records")
            data_source = 'aqi_features'
        
        # Fallback to aqi_measurements
        elif 'aqi_measurements' in fs_db.list_collection_names():
            measurement_count = fs_db['aqi_measurements'].count_documents({})
            print(f"   ✅ Found {measurement_count:,} measurement records")
            data_source = 'aqi_measurements'
        
        else:
            print("❌ No data found for EDA")
            print("💡 Please run data collection first")
            mongo_manager.log_pipeline_step('eda', 'skipped', {
                'reason': 'No data available',
                'parent_log_id': log_id
            })
            return
        
        # Step 2: Run YOUR EDA script
        print("\n2️⃣ RUNNING EDA SCRIPT")
        print("-" * 40)
        
        result = run_your_eda()
        
        if not result or not result.get('success', False):
            print("❌ EDA analysis failed")
            mongo_manager.log_pipeline_step('eda', 'failed', {
                'error': result.get('error', 'Unknown error') if result else 'No result',
                'parent_log_id': log_id
            })
            return
        
        # Step 3: Check EDA results
        print("\n3️⃣ CHECKING EDA RESULTS")
        print("-" * 40)
        
        eda_results = check_eda_results()
        
        total_files = len(eda_results['plots']) + len(eda_results['reports']) + len(eda_results['statistics'])
        print(f"   📁 Total files generated: {total_files}")
        
        if total_files == 0:
            print("⚠️  No EDA files were generated")
            print("💡 Check if the results/ directory was created")
        
        # Step 4: Store EDA metadata in MongoDB
        print("\n4️⃣ STORING EDA METADATA")
        print("-" * 40)
        
        # Store in pipeline logs database
        pl_db = mongo_manager.client[mongo_manager.pipeline_logs_db]
        
        eda_doc = {
            'timestamp': datetime.utcnow(),
            'data_source': data_source,
            'plots_generated': len(eda_results['plots']),
            'reports_generated': len(eda_results['reports']),
            'statistics_files': len(eda_results['statistics']),
            'total_files': total_files,
            'insights_count': result.get('insights_count', 0),
            'run_type': 'github_actions' if os.getenv('GITHUB_ACTIONS') == 'true' else 'local',
            'city': 'Karachi'
        }
        
        pl_db['eda_reports'].insert_one(eda_doc)
        print(f"   💾 EDA metadata stored in MongoDB")
        
        # Step 5: Log success
        print("\n5️⃣ LOGGING EXECUTION")
        print("-" * 40)
        
        mongo_manager.log_pipeline_step('eda', 'completed', {
            'success': True,
            'insights_count': result.get('insights_count', 0),
            'plots_generated': len(eda_results['plots']),
            'reports_generated': len(eda_results['reports']),
            'total_files': total_files,
            'output_lines': result.get('output_lines', 0),
            'parent_log_id': log_id,
            'city': 'Karachi'
        })
        
        print("\n" + "=" * 60)
        print("✅ EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ EDA pipeline failed: {str(e)}")
        traceback.print_exc()
        
        mongo_manager.log_pipeline_step('eda', 'failed', {
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