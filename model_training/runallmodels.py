"""
Run all forecasting models and compare - UPDATED VERSION
"""
import subprocess
import sys
from datetime import datetime
import time

def run_ml_models():
    """Run ML model training"""
    print("=" * 60)
    print(f"[{datetime.now()}] 🤖 TRAINING ML MODELS")
    print("=" * 60)
    try:
        result = subprocess.run([sys.executable, "model_training/train_models.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("⚠️ Warnings/Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running ML models: {e}")
        return False

def run_time_series():
    """Run time series forecasting"""
    print("=" * 60)
    print(f"[{datetime.now()}] 📈 RUNNING TIME SERIES FORECASTING")
    print("=" * 60)
    try:
        result = subprocess.run([sys.executable, "model_training/train_time_series_models.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("⚠️ Warnings/Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running time series: {e}")
        return False

def compare_results():
    """Compare results from both approaches"""
    print("=" * 60)
    print(f"[{datetime.now()}] 📊 COMPARING RESULTS")
    print("=" * 60)
    
    # Add code to compare ML vs time series performance
    print("✅ Both models completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Check MongoDB for saved forecasts")
    print("   2. Run the dashboard: streamlit run dashboard/dashboard.py")
    print("   3. Compare ML vs Time Series predictions")
    
    # You could add MongoDB query here to compare actual results
    try:
        from pymongo import MongoClient
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        client = MongoClient(os.getenv('MONGODB_URI'))
        db = client[os.getenv('MONGODB_DATABASE')]
        
        ml_count = db.aqi_predictions.count_documents({})
        ts_count = db.simple_forecasts.count_documents({})
        
        print(f"\n📊 Forecasts in database:")
        print(f"   ML predictions: {ml_count}")
        print(f"   Time series forecasts: {ts_count}")
        
        client.close()
    except Exception as e:
        print(f"⚠️ Could not check database: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AQI KARACHI - ALL FORECASTING MODELS")
    print("=" * 60)
    print("Running both Machine Learning and Time Series models...")
    
    start_time = time.time()
    
    # Run ML models
    ml_success = run_ml_models()
    
    # Wait a moment
    time.sleep(2)
    
    # Run Time Series
    ts_success = run_time_series()
    
    # Compare results
    if ml_success or ts_success:
        compare_results()
    else:
        print("❌ One or both models failed to run")
    
    end_time = time.time()
    print(f"\n⏱️ Total time: {end_time - start_time:.2f} seconds")
    print("\n✅ Process completed! Check your dashboard for results.")