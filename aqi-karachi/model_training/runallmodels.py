"""
Run all forecasting models and compare - UPDATED VERSION
Uses the new combined training script for 3-day forecasts
"""
import subprocess
import sys
from datetime import datetime
import time
import os

def run_combined_training():
    """Run combined ML + Time Series training"""
    print("=" * 70)
    print(f"[{datetime.now()}] 🚀 RUNNING COMBINED TRAINING (3-DAY FORECAST)")
    print("=" * 70)
    
    try:
        # Run the combined training script
        result = subprocess.run(
            [sys.executable, "model_training/combinedtraining.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"❌ Combined training failed with exit code: {result.returncode}")
            if result.stderr:
                print("Error details:")
                print(result.stderr[:500])  # Print first 500 chars of error
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Combined training timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running combined training: {e}")
        return False

def run_individual_models():
    """Run individual models (backup option)"""
    print("=" * 60)
    print(f"[{datetime.now()}] 🤖 RUNNING INDIVIDUAL MODELS")
    print("=" * 60)
    
    success_flags = {'ml': False, 'time_series': False}
    
    # Run ML models
    print("\n[1/2] Running ML model training...")
    try:
        ml_result = subprocess.run(
            [sys.executable, "model_training/train_models.py"],
            capture_output=True,
            text=True
        )
        print(ml_result.stdout[:500])  # Print first 500 chars
        success_flags['ml'] = ml_result.returncode == 0
    except Exception as e:
        print(f"❌ ML model failed: {e}")
    
    # Wait a moment
    time.sleep(2)
    
    # Run Time Series
    print("\n[2/2] Running time series forecasting...")
    try:
        ts_result = subprocess.run(
            [sys.executable, "model_training/train_time_series_models.py"],
            capture_output=True,
            text=True
        )
        print(ts_result.stdout[:500])  # Print first 500 chars
        success_flags['time_series'] = ts_result.returncode == 0
    except Exception as e:
        print(f"❌ Time series failed: {e}")
    
    return success_flags

def check_results():
    """Check and display results"""
    print("=" * 60)
    print(f"[{datetime.now()}] 📊 CHECKING RESULTS")
    print("=" * 60)
    
    # Check for reports
    reports_dir = 'reports'
    if os.path.exists(reports_dir):
        report_files = [f for f in os.listdir(reports_dir) if f.endswith(('.md', '.json'))]
        if report_files:
            print(f"✅ Found {len(report_files)} report files:")
            for f in sorted(report_files)[-3:]:  # Show last 3 reports
                print(f"   📄 {f}")
            
            # Read and display latest report summary
            try:
                md_reports = [f for f in report_files if f.endswith('.md')]
                if md_reports:
                    latest_md = max([os.path.join(reports_dir, f) for f in md_reports], 
                                   key=os.path.getctime)
                    
                    with open(latest_md, 'r') as f:
                        content = f.read()
                    
                    # Extract 3-day forecast section
                    if "3-DAY FORECAST SUMMARY" in content:
                        print("\n📊 Latest 3-day forecast summary:")
                        lines = content.split('\n')
                        summary_start = None
                        for i, line in enumerate(lines):
                            if "3-DAY FORECAST SUMMARY" in line:
                                summary_start = i
                                break
                        
                        if summary_start and summary_start + 5 < len(lines):
                            for j in range(summary_start, min(summary_start + 10, len(lines))):
                                print(f"   {lines[j]}")
            except Exception as e:
                print(f"⚠️ Could not read report: {e}")
        else:
            print("❌ No reports found")
    else:
        print("❌ Reports directory not found")
    
    # Check models directory
    models_dir = 'models'
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pkl', '.joblib'))]
        print(f"\n🤖 Found {len(model_files)} trained model files")
    
    print("\n📋 Next steps:")
    print("   1. Check 'reports/' directory for forecast reports")
    print("   2. Check MongoDB for saved forecasts")
    print("   3. Run dashboard: streamlit run dashboard/dashboard.py")

def main():
    print("=" * 70)
    print("🚀 AQI KARACHI - ALL FORECASTING MODELS")
    print("=" * 70)
    print("Options:")
    print("  1. Run Combined Training (Recommended)")
    print("  2. Run Individual Models")
    print("  3. Exit")
    print("=" * 70)
    
    try:
        choice = input("Enter choice (1, 2, or 3): ").strip()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
        return
    
    if choice == "1":
        print("\n" + "=" * 70)
        print("🚀 STARTING COMBINED TRAINING PIPELINE")
        print("=" * 70)
        
        start_time = time.time()
        
        # Run combined training
        success = run_combined_training()
        
        end_time = time.time()
        
        if success:
            check_results()
            print(f"\n✅ Combined training completed in {end_time - start_time:.2f} seconds")
        else:
            print(f"\n❌ Combined training failed after {end_time - start_time:.2f} seconds")
            print("💡 Trying individual models as backup...")
            
            # Fallback to individual models
            time.sleep(2)
            success_flags = run_individual_models()
            if any(success_flags.values()):
                check_results()
    
    elif choice == "2":
        print("\n" + "=" * 70)
        print("🤖 STARTING INDIVIDUAL MODEL TRAINING")
        print("=" * 70)
        
        start_time = time.time()
        
        # Run individual models
        success_flags = run_individual_models()
        
        end_time = time.time()
        
        if any(success_flags.values()):
            check_results()
            print(f"\n✅ Individual models completed in {end_time - start_time:.2f} seconds")
            print(f"   ML: {'✅' if success_flags['ml'] else '❌'}")
            print(f"   Time Series: {'✅' if success_flags['time_series'] else '❌'}")
        else:
            print(f"\n❌ All models failed after {end_time - start_time:.2f} seconds")
    
    elif choice == "3":
        print("\n👋 Exiting...")
        return
    
    else:
        print(f"\n❌ Invalid choice: {choice}")
        print("Please run the script again with 1, 2, or 3")
        return
    
    print("\n" + "=" * 70)
    print("🎉 PROCESS COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()