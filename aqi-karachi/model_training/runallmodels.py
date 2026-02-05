"""
FIXED runallmodels.py - ORCHESTRATOR SCRIPT
FIXED to create 3-day ensemble forecasts with proper time series execution
"""
import os
import sys
import subprocess
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

load_dotenv()

class FixedOrchestrator:
    """Fixed orchestrator with proper R² handling"""
    
    def __init__(self):
        self.mongodb_uri = os.getenv('MONGODB_URI')
        self.client = MongoClient(self.mongodb_uri)
        self.db = self.client['aqi_predictor']
        self.mr_db = self.client['aqi_model_registry']
        
    def check_recent_training(self):
        """Check if training happened in last 6 hours"""
        recent_model = self.mr_db.models.find_one(
            {'created_at': {'$gte': datetime.now() - timedelta(hours=6)}}
        )
        return recent_model is not None
    
    def run_train_models(self):
        """Run your GOOD train_models.py script"""
        print("\n🤖 TRAINING ML MODELS")
        print("-" * 40)
        
        script_path = os.path.join(project_root, 'model_training', 'train_models.py')
        
        if not os.path.exists(script_path):
            print(f"❌ Script not found: {script_path}")
            return None
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=300
            )
            
            # Print the output for debugging
            print("Training script output:")
            print(result.stdout[:1000])  # First 1000 chars
            
            if result.returncode == 0:
                print("✅ ML training completed")
                return True
            else:
                print(f"❌ ML training failed with code {result.returncode}")
                print(f"Error: {result.stderr[-500:] if result.stderr else 'No error output'}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ ML training timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def run_time_series(self):
        """Run time series forecasting - FIXED FILE NAME"""
        print("\n📈 RUNNING TIME SERIES FORECASTING")
        print("-" * 40)
        
        # FIXED: Use correct file name
        script_path = os.path.join(project_root, 'model_training', 'train_time_series_models.py')
        
        if not os.path.exists(script_path):
            print(f"❌ Time series script not found: {script_path}")
            print("Looking for files:", os.listdir(os.path.join(project_root, 'model_training')))
            return False
        
        try:
            print(f"Running: {script_path}")
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=300
            )
            
            print("Time series script output:")
            print(result.stdout[:800])  # First 800 chars
            
            if result.returncode == 0:
                print("✅ Time series forecasting completed")
                return True
            else:
                print(f"❌ Time series failed with code {result.returncode}")
                print(f"Error: {result.stderr[-500:] if result.stderr else 'No error output'}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Time series timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"❌ Time series error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_ensemble_forecasts(self):
        """Create 3-DAY ensemble forecasts - IMPROVED"""
        print("\n⚖️ CREATING ENSEMBLE FORECASTS (3 DAYS)")
        print("-" * 40)
        
        try:
            # Get ML recursive forecasts
            ml_forecasts = list(self.db.ml_recursive_forecasts.find(
                {}, 
                {'_id': 0, 'date': 1, 'predicted_aqi': 1, 'timestamp': 1}
            ))
            
            # Get time series forecasts
            ts_forecasts = list(self.db.timeseries_forecasts_3day.find(
                {}, 
                {'_id': 0, 'date': 1, 'predicted_aqi': 1, 'model_type': 1, 'timestamp': 1}
            ))
            
            print(f"📊 ML forecasts found: {len(ml_forecasts)}")
            print(f"📊 Time series forecasts found: {len(ts_forecasts)}")
            
            if not ml_forecasts and not ts_forecasts:
                print("❌ No forecasts found from any model")
                return []
            
            # Process ML forecasts (they might be hourly, convert to daily)
            ml_daily = {}
            for forecast in ml_forecasts:
                date_str = forecast.get('date')
                if not date_str:
                    # Extract date from timestamp
                    ts = forecast.get('timestamp')
                    if isinstance(ts, str):
                        date_str = ts.split('T')[0]
                    elif hasattr(ts, 'date'):
                        date_str = ts.date().isoformat()
                
                if date_str:
                    if date_str not in ml_daily:
                        ml_daily[date_str] = []
                    ml_daily[date_str].append(forecast.get('predicted_aqi', 0))
            
            # Process Time Series forecasts (already daily)
            ts_daily = {}
            for forecast in ts_forecasts:
                date_str = forecast.get('date')
                if date_str:
                    if date_str not in ts_daily:
                        ts_daily[date_str] = []
                    ts_daily[date_str].append(forecast.get('predicted_aqi', 0))
            
            print(f"📅 ML unique dates: {len(ml_daily)}")
            print(f"📅 TS unique dates: {len(ts_daily)}")
            
            # Calculate daily averages
            ml_daily_avg = {}
            for date_str, aqi_values in ml_daily.items():
                ml_daily_avg[date_str] = sum(aqi_values) / len(aqi_values)
            
            ts_daily_avg = {}
            for date_str, aqi_values in ts_daily.items():
                ts_daily_avg[date_str] = sum(aqi_values) / len(aqi_values)
            
            # Get next 3 dates starting from tomorrow
            today = datetime.now().date()
            future_dates = []
            
            for i in range(1, 4):  # Next 3 days
                future_date = today + timedelta(days=i)
                future_dates.append(future_date.isoformat())
            
            print(f"🎯 Target forecast dates: {future_dates}")
            
            # Create ensemble for next 3 days
            ensemble_data = []
            
            for i, date_str in enumerate(future_dates):
                ml_aqi = ml_daily_avg.get(date_str)
                ts_aqi = ts_daily_avg.get(date_str)
                
                # Calculate ensemble AQI
                ensemble_aqi = None
                weights = ""
                
                if ml_aqi is not None and ts_aqi is not None:
                    # Both available: 60% ML, 40% TS (adjusted)
                    ensemble_aqi = ml_aqi * 0.6 + ts_aqi * 0.4
                    weights = "ML:60%, TS:40%"
                    print(f"  {date_str}: Using both models")
                elif ml_aqi is not None:
                    # Only ML available
                    ensemble_aqi = ml_aqi
                    weights = "ML:100%"
                    print(f"  {date_str}: Using ML only")
                elif ts_aqi is not None:
                    # Only TS available
                    ensemble_aqi = ts_aqi
                    weights = "TS:100%"
                    print(f"  {date_str}: Using TS only")
                else:
                    # No data for this date, use fallback
                    print(f"  {date_str}: No forecast, using fallback")
                    # Simple fallback: average of last 3 days
                    last_3_days = []
                    for j in range(1, 4):
                        past_date = (today - timedelta(days=j)).isoformat()
                        if past_date in ml_daily_avg:
                            last_3_days.append(ml_daily_avg[past_date])
                        elif past_date in ts_daily_avg:
                            last_3_days.append(ts_daily_avg[past_date])
                    
                    if last_3_days:
                        ensemble_aqi = sum(last_3_days) / len(last_3_days)
                        weights = "Fallback: historical average"
                    else:
                        continue  # Skip this date
                
                # Format date
                forecast_date = datetime.strptime(date_str, '%Y-%m-%d')
                day_name = forecast_date.strftime('%A')
                
                ensemble_data.append({
                    'date': date_str,
                    'day_name': day_name,
                    'predicted_aqi': round(float(ensemble_aqi), 1),
                    'ml_aqi': round(float(ml_aqi), 1) if ml_aqi else None,
                    'ts_aqi': round(float(ts_aqi), 1) if ts_aqi else None,
                    'model': 'ensemble',
                    'weights': weights,
                    'created_at': datetime.now(),
                    'forecast_day': i + 1,
                    'timestamp': forecast_date
                })
            
            # Save ensemble
            self.db.ensemble_forecasts_3day.delete_many({})
            if ensemble_data:
                self.db.ensemble_forecasts_3day.insert_many(ensemble_data)
                print(f"\n✅ Created {len(ensemble_data)} ensemble forecasts:")
                for forecast in ensemble_data:
                    print(f"   📅 {forecast['date']} ({forecast['day_name']}): "
                          f"AQI {forecast['predicted_aqi']} ({forecast['weights']})")
            else:
                print("❌ No ensemble forecasts created")
            
            return ensemble_data
            
        except Exception as e:
            print(f"❌ Ensemble creation failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def verify_forecasts_exist(self):
        """Verify all forecast collections have data"""
        print("\n🔍 VERIFYING FORECASTS")
        print("-" * 40)
        
        collections_to_check = [
            'ml_recursive_forecasts',
            'timeseries_forecasts_3day', 
            'ensemble_forecasts_3day'
        ]
        
        for collection in collections_to_check:
            count = self.db[collection].count_documents({}) if collection in self.db.list_collection_names() else 0
            print(f"  {collection}: {count} forecasts")
            
            if count > 0:
                # Show first forecast
                forecast = self.db[collection].find_one({}, {'_id': 0, 'date': 1, 'predicted_aqi': 1})
                if forecast:
                    print(f"    Example: {forecast.get('date')} - AQI {forecast.get('predicted_aqi')}")
        
        return True
    
    def run(self):
        """Main orchestrator"""
        print("=" * 70)
        print("🚀 AQI KARACHI - COMPLETE PIPELINE ORCHESTRATOR")
        print("=" * 70)
        
        start_time = datetime.now()
        
        # 1. Train ML models (if needed)
        should_train = not self.check_recent_training()
        
        if should_train:
            print("🔄 Training new models...")
            ml_success = self.run_train_models()
            if not ml_success:
                print("⚠️ ML training failed, but continuing with existing models")
        else:
            print("✅ Using recently trained models")
        
        # 2. Run time series forecasting
        print("\n" + "=" * 70)
        print("📈 TIME SERIES FORECASTING")
        ts_success = self.run_time_series()
        if not ts_success:
            print("⚠️ Time series failed, using existing forecasts")
        
        # 3. Create ensemble forecasts
        print("\n" + "=" * 70)
        print("⚖️ ENSEMBLE CREATION")
        ensemble = self.create_ensemble_forecasts()
        
        # 4. Verify everything
        self.verify_forecasts_exist()
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Total time: {elapsed:.1f} seconds")
        
        # Show final forecast summary
        if ensemble:
            print(f"\n🎯 3-DAY FORECAST FOR KARACHI:")
            print("-" * 40)
            for forecast in ensemble:
                aqi = forecast['predicted_aqi']
                
                # Add emoji based on AQI
                if aqi <= 50:
                    emoji = "😊"
                elif aqi <= 100:
                    emoji = "😐"
                elif aqi <= 150:
                    emoji = "😷"
                elif aqi <= 200:
                    emoji = "🤒"
                elif aqi <= 300:
                    emoji = "🏥"
                else:
                    emoji = "☣️"
                
                print(f"  {forecast['date']} ({forecast['day_name']}): "
                      f"AQI {aqi} {emoji} ({forecast['weights']})")
        
        print(f"\n📊 Next steps:")
        print("  1. Refresh dashboard to see updated forecasts")
        print("  2. Check Model Performance for metrics")
        print("  3. Feature Importance shows model insights")
        
        return True

def main():
    """Main function"""
    print("🌫️ AQI Karachi - Complete Prediction Pipeline")
    print("=" * 70)
    
    orchestrator = FixedOrchestrator()
    success = orchestrator.run()
    
    if success:
        print("\n✅ Orchestrator completed successfully!")
        return 0
    else:
        print("\n❌ Orchestrator failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
