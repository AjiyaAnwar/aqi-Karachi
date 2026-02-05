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

# Add project root to path - FIXED PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # aqi-karachi directory
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
            return False
        
        try:
            print(f"📄 Running: {script_path}")
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=300
            )
            
            # Print the output for debugging
            print("Training script output (first 500 chars):")
            print(result.stdout[:500] if result.stdout else "No output")
            
            if result.returncode == 0:
                print("✅ ML training completed")
                return True
            else:
                print(f"❌ ML training failed with code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr[-200:]}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ ML training timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def run_time_series(self):
        """Run time series forecasting - FIXED FILE PATH"""
        print("\n📈 RUNNING TIME SERIES FORECASTING")
        print("-" * 40)
        
        # FIXED: Use correct file path
        script_path = os.path.join(project_root, 'model_training', 'train_time_series_models.py')
        
        if not os.path.exists(script_path):
            print(f"❌ Time series script not found: {script_path}")
            print("📁 Looking for files in model_training directory:")
            model_training_dir = os.path.join(project_root, 'model_training')
            if os.path.exists(model_training_dir):
                print(f"Files: {os.listdir(model_training_dir)}")
            else:
                print(f"Directory doesn't exist: {model_training_dir}")
            return False
        
        try:
            print(f"📄 Running time series script: {os.path.basename(script_path)}")
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=300
            )
            
            print("Time series script output (first 500 chars):")
            if result.stdout:
                print(result.stdout[:500])
            
            if result.returncode == 0:
                print("✅ Time series forecasting completed")
                # Check if forecasts were actually created
                ts_count = self.db.timeseries_forecasts_3day.count_documents({})
                print(f"📊 Time series forecasts in DB: {ts_count}")
                return True
            else:
                print(f"❌ Time series failed with code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr[-200:]}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Time series timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"❌ Time series error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_time_series_forecasts(self):
        """Check if time series forecasts exist and are fresh"""
        try:
            # Check for time series forecasts
            ts_count = self.db.timeseries_forecasts_3day.count_documents({})
            
            if ts_count > 0:
                # Check when they were created
                latest_ts = self.db.timeseries_forecasts_3day.find_one(
                    sort=[('created_at', -1)]
                )
                if latest_ts:
                    created_at = latest_ts.get('created_at')
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at.replace('Z', ''))
                    
                    age_hours = (datetime.now() - created_at).total_seconds() / 3600
                    
                    if age_hours < 24:
                        print(f"✅ Time series forecasts exist ({ts_count} records, {age_hours:.1f} hours old)")
                        return True
                    else:
                        print(f"⚠️ Time series forecasts are old ({age_hours:.1f} hours)")
                        return False
            
            print(f"⚠️ No time series forecasts found or empty ({ts_count} records)")
            return False
            
        except Exception as e:
            print(f"❌ Error checking time series: {e}")
            return False
    
    def create_ensemble_forecasts(self):
        """Create 3-DAY ensemble forecasts"""
        print("\n⚖️ CREATING ENSEMBLE FORECASTS (3 DAYS)")
        print("-" * 40)
        
        try:
            # Check what forecasts we have
            ml_count = self.db.ml_recursive_forecasts.count_documents({})
            ts_count = self.db.timeseries_forecasts_3day.count_documents({})
            
            print(f"📊 ML forecasts available: {ml_count}")
            print(f"📊 Time series forecasts available: {ts_count}")
            
            if ml_count == 0 and ts_count == 0:
                print("❌ No forecasts available from any model")
                return []
            
            # Get ML forecasts
            ml_forecasts = []
            if ml_count > 0:
                ml_forecasts = list(self.db.ml_recursive_forecasts.find(
                    {}, 
                    {'_id': 0, 'date': 1, 'predicted_aqi': 1, 'timestamp': 1}
                ))
                print(f"📅 ML forecasts loaded: {len(ml_forecasts)}")
            
            # Get time series forecasts
            ts_forecasts = []
            if ts_count > 0:
                ts_forecasts = list(self.db.timeseries_forecasts_3day.find(
                    {}, 
                    {'_id': 0, 'date': 1, 'predicted_aqi': 1, 'model_type': 1, 'timestamp': 1}
                ))
                print(f"📅 Time series forecasts loaded: {len(ts_forecasts)}")
            
            # Process ML forecasts to get daily averages
            ml_daily = {}
            for forecast in ml_forecasts:
                date_str = forecast.get('date')
                if not date_str:
                    # Extract date from timestamp
                    ts = forecast.get('timestamp')
                    if isinstance(ts, str):
                        try:
                            date_str = ts.split('T')[0]
                        except:
                            continue
                    elif hasattr(ts, 'date'):
                        date_str = ts.date().isoformat()
                    else:
                        continue
                
                if date_str not in ml_daily:
                    ml_daily[date_str] = []
                ml_daily[date_str].append(forecast.get('predicted_aqi', 0))
            
            # Process Time Series forecasts
            ts_daily = {}
            for forecast in ts_forecasts:
                date_str = forecast.get('date')
                if date_str:
                    if date_str not in ts_daily:
                        ts_daily[date_str] = []
                    ts_daily[date_str].append(forecast.get('predicted_aqi', 0))
            
            # Calculate daily averages
            ml_daily_avg = {}
            for date_str, aqi_values in ml_daily.items():
                ml_daily_avg[date_str] = sum(aqi_values) / len(aqi_values)
            
            ts_daily_avg = {}
            for date_str, aqi_values in ts_daily.items():
                ts_daily_avg[date_str] = sum(aqi_values) / len(aqi_values)
            
            print(f"📅 ML unique dates: {len(ml_daily_avg)}")
            print(f"📅 TS unique dates: {len(ts_daily_avg)}")
            
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
                
                print(f"\n📅 Processing {date_str}:")
                print(f"   ML AQI: {ml_aqi}")
                print(f"   TS AQI: {ts_aqi}")
                
                # Calculate ensemble AQI
                ensemble_aqi = None
                weights = ""
                
                if ml_aqi is not None and ts_aqi is not None:
                    # Both available: 60% ML, 40% TS
                    ensemble_aqi = ml_aqi * 0.6 + ts_aqi * 0.4
                    weights = "ML:60%, TS:40%"
                    print(f"   Using both models → {ensemble_aqi:.1f}")
                elif ml_aqi is not None:
                    # Only ML available
                    ensemble_aqi = ml_aqi
                    weights = "ML:100%"
                    print(f"   Using ML only → {ensemble_aqi:.1f}")
                elif ts_aqi is not None:
                    # Only TS available
                    ensemble_aqi = ts_aqi
                    weights = "TS:100%"
                    print(f"   Using TS only → {ensemble_aqi:.1f}")
                else:
                    # No data for this date
                    print(f"   No forecast available")
                    
                    # Try to get any forecast for this date (maybe from earlier runs)
                    all_forecasts = self.db.ensemble_forecasts_3day.find_one(
                        {'date': date_str},
                        {'_id': 0, 'predicted_aqi': 1}
                    )
                    
                    if all_forecasts and 'predicted_aqi' in all_forecasts:
                        ensemble_aqi = all_forecasts['predicted_aqi']
                        weights = "From previous run"
                        print(f"   Using previous forecast → {ensemble_aqi}")
                    else:
                        # Skip this date
                        print(f"   Skipping {date_str} - no forecast available")
                        continue
                
                # Format date
                forecast_date = datetime.strptime(date_str, '%Y-%m-%d')
                day_name = forecast_date.strftime('%A')
                
                # Categorize AQI
                if ensemble_aqi <= 50:
                    category = "Good"
                elif ensemble_aqi <= 100:
                    category = "Moderate"
                elif ensemble_aqi <= 150:
                    category = "Unhealthy for Sensitive Groups"
                elif ensemble_aqi <= 200:
                    category = "Unhealthy"
                elif ensemble_aqi <= 300:
                    category = "Very Unhealthy"
                else:
                    category = "Hazardous"
                
                ensemble_data.append({
                    'date': date_str,
                    'day_name': day_name,
                    'predicted_aqi': round(float(ensemble_aqi), 1),
                    'category': category,
                    'ml_aqi': round(float(ml_aqi), 1) if ml_aqi else None,
                    'ts_aqi': round(float(ts_aqi), 1) if ts_aqi else None,
                    'model': 'ensemble',
                    'weights': weights,
                    'created_at': datetime.now(),
                    'forecast_day': i + 1,
                    'timestamp': forecast_date
                })
            
            # Save ensemble
            if ensemble_data:
                self.db.ensemble_forecasts_3day.delete_many({})
                self.db.ensemble_forecasts_3day.insert_many(ensemble_data)
                print(f"\n✅ Created {len(ensemble_data)} ensemble forecasts:")
                for forecast in ensemble_data:
                    print(f"   📅 {forecast['date']} ({forecast['day_name']}): "
                          f"AQI {forecast['predicted_aqi']} - {forecast['category']} ({forecast['weights']})")
            else:
                print("❌ No ensemble forecasts created")
                # Try to keep existing forecasts
                existing = self.db.ensemble_forecasts_3day.count_documents({})
                if existing > 0:
                    print(f"⚠️ Keeping existing {existing} ensemble forecasts")
            
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
        
        all_good = True
        for collection in collections_to_check:
            count = self.db[collection].count_documents({}) if collection in self.db.list_collection_names() else 0
            status = "✅" if count > 0 else "❌"
            print(f"  {status} {collection}: {count} forecasts")
            
            if count > 0:
                # Show first forecast
                forecast = self.db[collection].find_one({}, {'_id': 0, 'date': 1, 'predicted_aqi': 1})
                if forecast:
                    print(f"    Example: {forecast.get('date')} - AQI {forecast.get('predicted_aqi')}")
            else:
                all_good = False
        
        return all_good
    
    def run(self):
        """Main orchestrator"""
        print("=" * 70)
        print("🚀 AQI KARACHI - COMPLETE PIPELINE ORCHESTRATOR")
        print("=" * 70)
        
        start_time = datetime.now()
        
        # Check if we need to run ML training
        should_train = not self.check_recent_training()
        
        if should_train:
            print("🔄 Training new models...")
            ml_success = self.run_train_models()
            if not ml_success:
                print("⚠️ ML training failed, but continuing with existing models")
        else:
            print("✅ Using recently trained models")
        
        # Check if time series forecasts exist and are fresh
        ts_fresh = self.check_time_series_forecasts()
        
        if not ts_fresh:
            print("\n🔄 Running time series forecasting...")
            ts_success = self.run_time_series()
            if not ts_success:
                print("⚠️ Time series failed, trying to use existing forecasts")
        else:
            print("\n✅ Using existing time series forecasts")
        
        # Create ensemble forecasts
        print("\n" + "=" * 70)
        ensemble = self.create_ensemble_forecasts()
        
        # Verify everything
        print("\n" + "=" * 70)
        all_good = self.verify_forecasts_exist()
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE EXECUTION COMPLETE")
        print("=" * 70)
        print(f"Total time: {elapsed:.1f} seconds")
        
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
                      f"AQI {aqi} {emoji} ({forecast['category']})")
        
        print(f"\n📊 Next steps:")
        print("  1. Refresh dashboard to see updated forecasts")
        print("  2. Check Model Performance for metrics")
        print("  3. Feature Importance shows model insights")
        
        return all_good

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
        print("\n⚠️ Orchestrator completed with warnings")
        return 1

if __name__ == "__main__":
    sys.exit(main())