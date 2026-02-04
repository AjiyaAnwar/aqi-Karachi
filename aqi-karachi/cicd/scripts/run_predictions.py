#!/usr/bin/env python
"""
Prediction Runner for CI/CD
Runs the orchestrator to generate all forecasts
"""
import os
import sys
import subprocess
from pathlib import Path

# ==================== ENVIRONMENT SETUP ====================
project_root = Path(__file__).parent.parent.parent  # Go up 3 levels from cicd/scripts/
sys.path.append(str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded environment from: {env_path}")
else:
    load_dotenv()

def main():
    """Main prediction function"""
    print("=" * 60)
    print("🎯 PREDICTION RUNNER")
    print("📍 City: Karachi")
    print("=" * 60)
    
    # Check if orchestrator exists
    orchestrator_path = project_root / "model_training" / "runallmodels.py"
    
    if not orchestrator_path.exists():
        print(f"❌ Orchestrator not found: {orchestrator_path}")
        return False
    
    try:
        # Run orchestrator directly (no input needed)
        result = subprocess.run(
            [sys.executable, str(orchestrator_path)],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=600  # 10 minutes timeout
        )
        
        print("📊 Orchestrator output:")
        print("-" * 40)
        
        # Show key output
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            important_lines = []
            
            # Collect important lines
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line for keyword in 
                       ['✅', '❌', '🚀', '📊', '🤖', '📈', '⚖️', '🎯', '📅', '📌']):
                    important_lines.append(line)
                elif any(term in line_lower for term in 
                       ['r²', 'mae', 'rmse', 'completed', 'failed', 'forecast', 'model', 'training']):
                    important_lines.append(line)
            
            # Show first 20 important lines
            for line in important_lines[:20]:
                print(f"  {line}")
            
            if len(important_lines) > 20:
                print(f"  ... and {len(important_lines) - 20} more lines")
        
        print("-" * 40)
        
        # Check if successful
        if result.returncode == 0:
            # Look for success indicators in output
            stdout_lower = result.stdout.lower()
            if any(term in stdout_lower for term in ['completed', 'success', '✅']):
                print("✅ Predictions generated successfully!")
                return True
            else:
                print("⚠️ Process completed but no success indicator found")
                return True  # Still return True if exit code is 0
        else:
            print(f"❌ Predictions failed with code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Predictions timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running predictions: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ PREDICTION RUNNER COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ PREDICTION RUNNER FAILED!")
        print("=" * 60)
        sys.exit(1)
