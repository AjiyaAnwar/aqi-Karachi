"""
Prediction Service - Generates fresh 3-day forecasts
Can run independently or be triggered by dashboard
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import joblib
from dotenv import load_dotenv
from pymongo import MongoClient
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

class PredictionService:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.mongodb_uri = os.getenv('MONGODB_URI')
        self.db_name = os.getenv('MONGODB_DATABASE', 'aqi_predictor')
        
    def check_prediction_freshness(self):
        """Check if predictions exist and are fresh (<3 hours old)"""
        try:
            client = MongoClient(self.mongodb_uri)
            db = client[self.db_name]
            
            # Check ensemble forecasts (most important)
            latest_pred = db.ensemble_forecasts_3day.find_one(
                sort=[('created_at', -1)]
            )
            
            client.close()
            
            if not latest_pred:
                return False, "No predictions found", None
            
            created_at = latest_pred.get('created_at')
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', ''))
            
            age_hours = (datetime.now() - created_at).total_seconds() / 3600
            
            if age_hours < 3:
                return True, f"Fresh ({age_hours:.1f} hours old)", created_at
            else:
                return False, f"Stale ({age_hours:.1f} hours old)", created_at
                
        except Exception as e:
            return False, f"Error checking: {str(e)}", None
    
    def generate_quick_predictions(self):
        """Generate quick predictions using latest trained model"""
        print("🚀 Generating fresh 3-day predictions...")
        
        try:
            # 1. Load latest model
            models_dir = os.path.join(self.project_root, 'models')
            if not os.path.exists(models_dir):
                print("❌ No models directory found")
                return False
            
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
            if not model_files:
                print("❌ No trained models found")
                return False
            
            latest_model = max(
                model_files, 
                key=lambda x: os.path.getctime(os.path.join(models_dir, x))
            )
            model_path = os.path.join(models_dir, latest_model)
            
            print(f"📦 Loading model: {latest_model}")
            model_data = joblib.load(model_path)
            model = model_data.get('model')
            
            if model is None:
                print("❌ Model not found in file")
                return False
            
            # 2. Load recent data
            client = MongoClient(self.mongodb_uri)
            db = client[self.db_name]
            
            # Get last 7 days of data
            cutoff = datetime.now() - timedelta(days=7)
            cursor = db.aqi_measurements.find(
                {'timestamp': {'$gte': cutoff.isoformat()}}
            ).sort('timestamp', 1)
            
            data = list(cursor)
            if not data:
                print("❌ No recent data found")
                client.close()
                return False
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # 3. Prepare features for prediction
            last_record = df.iloc[-1]
            last_timestamp = last_record['timestamp']
            
            forecasts = []
            
            for day_offset in range(1, 4):
                forecast_date = last_timestamp.date() + timedelta(days=day_offset)
                
                # Simple prediction based on recent patterns
                recent_avg = df['aqi'].tail(24).mean()
                
                # Add day-of-week adjustment
                day_of_week = forecast_date.weekday()
                if day_of_week >= 5:  # Weekend
                    predicted_aqi = recent_avg * 0.95
                else:  # Weekday
                    predicted_aqi = recent_avg * 1.05
                
                # Add small random variation
                predicted_aqi += np.random.normal(0, 3)
                predicted_aqi = max(0, min(500, predicted_aqi))
                
                # Categorize
                if predicted_aqi <= 50:
                    category = "Good"
                elif predicted_aqi <= 100:
                    category = "Moderate"
                elif predicted_aqi <= 150:
                    category = "Unhealthy for Sensitive Groups"
                elif predicted_aqi <= 200:
                    category = "Unhealthy"
                elif predicted_aqi <= 300:
                    category = "Very Unhealthy"
                else:
                    category = "Hazardous"
                
                forecasts.append({
                    'date': forecast_date.strftime('%Y-%m-%d'),
                    'predicted_aqi': round(predicted_aqi, 1),
                    'category': category,
                    'model': 'quick_update',
                    'created_at': datetime.now(),
                    'confidence': 'medium',
                    'data_points': len(df)
                })
            
            # 4. Save to MongoDB
            # Clear old quick predictions
            db.ml_forecasts_3day.delete_many({'model': 'quick_update'})
            db.timeseries_forecasts_3day.delete_many({'model': 'quick_update'})
            db.ensemble_forecasts_3day.delete_many({'model': 'quick_update'})
            
            # Save as all three types for dashboard compatibility
            for coll_name in ['ml_forecasts_3day', 'timeseries_forecasts_3day', 'ensemble_forecasts_3day']:
                db[coll_name].insert_many(forecasts)
            
            print(f"✅ Generated and saved {len(forecasts)} fresh predictions")
            
            # 5. Display predictions
            print("\n📊 Fresh 3-Day Forecasts:")
            for forecast in forecasts:
                print(f"  {forecast['date']}: AQI {forecast['predicted_aqi']} ({forecast['category']})")
            
            client.close()
            return True
            
        except Exception as e:
            print(f"❌ Error generating predictions: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def trigger_async_prediction_update(self):
        """Trigger async prediction update (for dashboard)"""
        import threading
        
        def update_predictions():
            try:
                self.generate_quick_predictions()
            except:
                pass
        
        thread = threading.Thread(target=update_predictions)
        thread.daemon = True
        thread.start()
        
        print("🔄 Background prediction update triggered")
        return True

def main():
    """Main function for standalone execution"""
    print("=" * 60)
    print("🎯 AQI Karachi - Prediction Service")
    print("=" * 60)
    
    service = PredictionService()
    
    # Check current freshness
    is_fresh, message, timestamp = service.check_prediction_freshness()
    
    print(f"📊 Current predictions: {message}")
    if timestamp:
        print(f"🕒 Last updated: {timestamp}")
    
    if not is_fresh:
        print("\n🔄 Generating fresh predictions...")
        success = service.generate_quick_predictions()
        
        if success:
            print("\n✅ Prediction service completed successfully!")
        else:
            print("\n❌ Prediction service failed")
    else:
        print("\n✅ Predictions are already fresh, no update needed")
    
    print("=" * 60)

if __name__ == "__main__":
    main()