"""
CI/CD Pipeline for AQI Karachi
"""
import schedule
import time
import subprocess
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_data_collection():
    """Run data collection"""
    print(f"[{datetime.now()}] 📥 Collecting data...")
    subprocess.run([sys.executable, "data_pipeline/collect_historical.py"])

def run_feature_engineering():
    """Create features"""
    print(f"[{datetime.now()}] 🔧 Creating features...")
    subprocess.run([sys.executable, "data_pipeline/features.py"])

def run_model_training():
    """Train models"""
    print(f"[{datetime.now()}] 🤖 Training models...")
    subprocess.run([sys.executable, "model_training/runallmodels.py"])

def main():
    """Main pipeline"""
    print("=" * 60)
    print("🚀 AQI Karachi CI/CD Pipeline")
    print("=" * 60)
    
    # Schedule tasks
    schedule.every().day.at("00:00").do(run_data_collection)    # Daily data
    schedule.every().day.at("01:00").do(run_feature_engineering) # Daily features
    schedule.every(3).hours.do(run_model_training)              # Train every 3 hours
    
    print("✅ Scheduled:")
    print("   - Data collection: Daily at 00:00")
    print("   - Feature engineering: Daily at 01:00")
    print("   - Model training: Every 3 hours")
    print("\n🔄 Running initial tasks...")
    
    # Run initial tasks
    run_data_collection()
    run_feature_engineering()
    run_model_training()
    
    print("\n⏰ Pipeline running. Press Ctrl+C to stop.")
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Pipeline stopped.")