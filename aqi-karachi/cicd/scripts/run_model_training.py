#!/usr/bin/env python
"""
FIXED Model Training Runner for CI/CD
Runs the FIXED orchestrator (runallmodels.py)
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

def check_model_registry():
    """Check if good models exist in registry"""
    try:
        from pymongo import MongoClient
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return 0
        
        client = MongoClient(uri)
        mr_db = client['aqi_model_registry']
        
        # Check SINGLE collection
        if 'model_registry' in mr_db.list_collection_names():
            # Count models with positive R²
            good_models = mr_db.model_registry.count_documents({
                'metrics.r2_score': {'$gt': 0}
            })
            client.close()
            return good_models
        
        client.close()
        return 0
        
    except Exception as e:
        return 0

def run_orchestrator():
    """Run the FIXED runallmodels.py"""
    print("🤖 Running FIXED orchestrator...")
    
    orchestrator_path = project_root / 'model_training' / 'runallmodels.py'
    
    if not orchestrator_path.exists():
        print(f"❌ Orchestrator not found: {orchestrator_path}")
        return False
    
    try:
        # Run orchestrator
        result = subprocess.run(
            [sys.executable, str(orchestrator_path)],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=600
        )
        
        print("📊 Orchestrator output:")
        print("-" * 40)
        
        success = False
        
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            
            # Show important lines
            for line in lines:
                if any(keyword in line for keyword in 
                       ['✅', '❌', 'R²:', 'MAE:', 'COMPLETED', 'FAILED']):
                    print(f"  {line}")
            
            success = '✅ ORCHESTRATOR COMPLETED' in result.stdout
        
        print("-" * 40)
        
        if result.returncode == 0 and success:
            print("✅ Orchestrator completed successfully")
            return True
        else:
            print(f"❌ Orchestrator failed with code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running orchestrator: {e}")
        return False

def main():
    """Main CI/CD function"""
    print("=" * 70)
    print("🤖 CI/CD MODEL TRAINING - FIXED")
    print("📍 City: Karachi")
    print("🕒 Time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    # Check existing models
    print("\n🔍 Checking existing models...")
    existing_models = check_model_registry()
    
    if existing_models > 0:
        print(f"✅ Found {existing_models} good models in registry")
    
    # Run orchestrator
    print("\n🚀 Running model training pipeline...")
    success = run_orchestrator()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ CI/CD PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ CI/CD PIPELINE FAILED")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()