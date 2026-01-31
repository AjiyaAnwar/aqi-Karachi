"""
FIXED runallmodels.py - ORCHESTRATOR SCRIPT
FIXED to create 3-day ensemble forecasts
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
        """Check if training happened in last hour"""
        recent_model = self.mr_db.model_registry.find_one(
            {'created_at': {'$gte': datetime.now() - timedelta(hours=1)}}
        )
        return recent_model is not None
    
    def run_train_models(self):
        """Run your GOOD train_models.py script"""
        print("\n🤖 TRAINING 3H RECURSIVE MODEL")
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
            print(result.stdout[:500])  # First 500 chars
            
            if result.returncode == 0:
                print("✅ ML training completed")
                return True
            else:
                print(f"❌ ML training failed with code {result.returncode}")
                print(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def run_time_series(self):
        """Run time series forecasting"""
        print("\n📈 RUNNING TIME SERIES")
        print("-" * 40)
        
        script_path = os.path.join(project_root, 'model_training', 'time_series_fixed.py')
        
        if not os.path.exists(script_path):
            print("⚠️  Using existing time series forecasts")
            return True
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=300
            )
            
            if result.returncode == 0:
                print("✅ Time series completed")
                return True
            else:
                print("⚠️  Time series failed, using existing forecasts")
                return True
                
        except Exception as e:
            print(f"⚠️  Time series error: {e}, using existing forecasts")
            return True
    
    def create_ensemble_forecasts(self):
        """Create 3-DAY ensemble forecasts"""
        print("\n⚖️ CREATING ENSEMBLE FORECASTS (3 DAYS)")
        print("-" * 40)
        
        try:
            # Get ML recursive forecasts - Get DAILY averages
            ml_forecasts = list(self.db.ml_recursive_forecasts.find(
                {}, 
                {'_id': 0, 'date': 1, 'predicted_aqi': 1}
            ))
            
            # Get time series forecasts
            ts_forecasts = list(self.db.timeseries_forecasts_3day.find(
                {}, 
                {'_id': 0, 'date': 1, 'predicted_aqi': 1, 'model_type': 1}
            ))
            
            if not ml_forecasts:
                print("❌ No ML forecasts found")
                return []
            
            # Group ML forecasts by date to get daily averages
            ml_by_date = {}
            for forecast in ml_forecasts:
                date_str = forecast.get('date', '')
                if date_str:
                    if date_str not in ml_by_date:
                        ml_by_date[date_str] = []
                    ml_by_date[date_str].append(forecast.get('predicted_aqi', 0))
            
            # Calculate daily averages for ML
            ml_daily_avg = {}
            for date_str, aqi_values in ml_by_date.items():
                ml_daily_avg[date_str] = sum(aqi_values) / len(aqi_values)
            
            # Group TS forecasts by date
            ts_by_date = {}
            for forecast in ts_forecasts:
                date_str = forecast.get('date', '')
                if date_str:
                    if date_str not in ts_by_date:
                        ts_by_date[date_str] = []
                    ts_by_date[date_str].append(forecast.get('predicted_aqi', 0))
            
            # Calculate daily averages for TS (use ensemble models if available)
            ts_daily_avg = {}
            for date_str, aqi_values in ts_by_date.items():
                ts_daily_avg[date_str] = sum(aqi_values) / len(aqi_values)
            
            # Get unique dates from both forecasts
            all_dates = set(list(ml_daily_avg.keys()) + list(ts_daily_avg.keys()))
            all_dates = sorted(all_dates)  # Sort chronologically
            
            # Take next 3 dates (excluding today)
            today = datetime.now().date().isoformat()
            future_dates = [d for d in all_dates if d > today][:3]
            
            if len(future_dates) < 3:
                print(f"⚠️  Only {len(future_dates)} future dates available")
            
            # Create ensemble for next 3 days
            ensemble_data = []
            
            for i, date_str in enumerate(future_dates[:3]):
                ml_aqi = ml_daily_avg.get(date_str, None)
                ts_aqi = ts_daily_avg.get(date_str, None)
                
                # Calculate ensemble AQI
                ensemble_aqi = None
                weights = ""
                
                if ml_aqi is not None and ts_aqi is not None:
                    # Both available: 70% ML, 30% TS
                    ensemble_aqi = ml_aqi * 0.7 + ts_aqi * 0.3
                    weights = "ML:70%, TS:30%"
                elif ml_aqi is not None:
                    # Only ML available
                    ensemble_aqi = ml_aqi
                    weights = "ML:100%"
                elif ts_aqi is not None:
                    # Only TS available
                    ensemble_aqi = ts_aqi
                    weights = "TS:100%"
                else:
                    # No data for this date
                    continue
                
                # Create date object
                forecast_date = datetime.strptime(date_str, '%Y-%m-%d')
                
                ensemble_data.append({
                    'date': date_str,
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
                print(f"✅ Created {len(ensemble_data)} ensemble forecasts for:")
                for forecast in ensemble_data:
                    print(f"   - {forecast['date']}: AQI {forecast['predicted_aqi']} ({forecast['weights']})")
            
            return ensemble_data
            
        except Exception as e:
            print(f"❌ Ensemble creation failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def verify_model_registry(self):
        """Verify models are in correct registry"""
        print("\n🔍 VERIFYING MODEL REGISTRY")
        print("-" * 40)
        
        try:
            collections = self.mr_db.list_collection_names()
            print(f"Available collections: {collections}")
            
            if 'model_registry' in collections:
                models = list(self.mr_db.model_registry.find({}, {'_id': 0, 'model_name': 1, 'metrics.test_r2': 1}))
                print(f"Models in 'model_registry': {len(models)}")
                for model in models:
                    r2 = model.get('metrics', {}).get('test_r2', 'N/A')
                    print(f"  - {model.get('model_name')}: R² = {r2}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error verifying registry: {e}")
            return False
    
    def run(self):
        """Main orchestrator"""
        print("=" * 70)
        print("🚀 FIXED AQI Karachi Orchestrator")
        print("=" * 70)
        
        start_time = datetime.now()
        
        # Check if recent training exists
        if self.check_recent_training():
            print("✅ Recent training found (last hour)")
            print("📊 Loading existing forecasts...")
        else:
            print("🔄 Training new models...")
            
            # 1. Train ML model
            ml_success = self.run_train_models()
            if not ml_success:
                print("❌ Pipeline stopped: ML training failed")
                return False
        
        # 2. Verify registry
        self.verify_model_registry()
        
        # 3. Run time series
        self.run_time_series()
        
        # 4. Create 3-day ensemble
        ensemble = self.create_ensemble_forecasts()
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("✅ ORCHESTRATOR COMPLETED")
        print("=" * 70)
        print(f"Ensemble forecasts created: {len(ensemble)} days")
        print(f"Total time: {elapsed:.1f} seconds")
        print("\n🎯 Next steps:")
        print("1. Refresh your dashboard to see updated metrics")
        print("2. Feature Importance page will show actual model features")
        print("3. Model Performance page will show correct R² scores")
        
        return True

def main():
    """Main function"""
    print("🌫️ AQI Karachi - Orchestrator")
    print("=" * 70)
    
    orchestrator = FixedOrchestrator()
    success = orchestrator.run()
    
    if success:
        print("\n✅ Orchestrator completed successfully!")
    else:
        print("\n❌ Orchestrator failed!")

if __name__ == "__main__":
    main()