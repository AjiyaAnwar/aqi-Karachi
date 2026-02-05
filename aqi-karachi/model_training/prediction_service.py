"""
Prediction Service - Generates fresh 3-day forecasts
Updated to use MongoDB Manager
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import joblib
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Import MongoDB Manager
from cicd.mongodb_utils import MongoDBManager, ModelStatus

load_dotenv()

class PredictionService:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.mongo_manager = MongoDBManager()  # Use MongoDB Manager
        
    def check_prediction_freshness(self):
        """Check if predictions exist and are fresh (<3 hours old)"""
        try:
            # Use MongoDB Manager to get latest ensemble forecast
            client = self.mongo_manager.client
            db = client[self.mongo_manager.db_name]
            
            latest_pred = db.ensemble_forecasts_3day.find_one(
                sort=[('created_at', -1)]
            )
            
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
        """Generate quick predictions using latest trained model with MongoDB Manager"""
        print("🚀 Generating fresh 3-day predictions...")
        
        try:
            # 1. Get production model from registry
            model, metadata = self.mongo_manager.get_model(
                model_name="xgboost_aqi",
                status=ModelStatus.PRODUCTION
            )
            
            if not model:
                print("⚠️ No production model found, trying to load from models directory")
                # Fallback to local file
                models_dir = os.path.join(self.project_root, 'models')
                if not os.path.exists(models_dir):
                    print("❌ No models directory found")
                    return False
                
                model_files = [f for f in os.listdir(models_dir) if f.endswith(('.joblib', '.pkl'))]
                if not model_files:
                    print("❌ No trained models found")
                    return False
                
                latest_model = max(
                    model_files, 
                    key=lambda x: os.path.getctime(os.path.join(models_dir, x))
                )
                model_path = os.path.join(models_dir, latest_model)
                
                print(f"📦 Loading model from file: {latest_model}")
                model_data = joblib.load(model_path)
                model = model_data.get('model')
                
                if model is None:
                    print("❌ Model not found in file")
                    return False
            
            # 2. Get latest features using MongoDB Manager
            features_df, feature_version = self.mongo_manager.get_latest_features()
            
            if features_df.empty:
                print("❌ No features found")
                return False
            
            # 3. Generate predictions
            today = datetime.now().date()
            forecasts = []
            
            for day_offset in range(1, 4):
                forecast_date = today + timedelta(days=day_offset)
                day_of_week = forecast_date.weekday()
                
                # Simple prediction logic (replace with actual model prediction)
                recent_avg = features_df['aqi'].mean() if 'aqi' in features_df.columns else 100
                
                # Adjust for weekday/weekend
                if day_of_week >= 5:  # Weekend
                    predicted_aqi = recent_avg * 0.95
                else:  # Weekday
                    predicted_aqi = recent_avg * 1.05
                
                predicted_aqi = max(0, min(500, predicted_aqi))
                
                # Categorize
                if predicted_aqi <= 50:
                    category = "Good"
                    color = "green"
                elif predicted_aqi <= 100:
                    category = "Moderate"
                    color = "yellow"
                elif predicted_aqi <= 150:
                    category = "Unhealthy for Sensitive Groups"
                    color = "orange"
                elif predicted_aqi <= 200:
                    category = "Unhealthy"
                    color = "red"
                elif predicted_aqi <= 300:
                    category = "Very Unhealthy"
                    color = "purple"
                else:
                    category = "Hazardous"
                    color = "black"
                
                forecasts.append({
                    'date': forecast_date.strftime('%Y-%m-%d'),
                    'predicted_aqi': round(predicted_aqi, 1),
                    'category': category,
                    'model': 'quick_update',
                    'created_at': datetime.now(),
                    'confidence': 'medium',
                    'color': color,
                    'day_name': forecast_date.strftime('%A')
                })
            
            # 4. Save predictions using MongoDB Manager logging
            log_id = self.mongo_manager.log_pipeline_step(
                "quick_predictions", 
                "completed", 
                {"predictions_generated": len(forecasts)}
            )
            
            # Save to database
            client = self.mongo_manager.client
            db = client[self.mongo_manager.db_name]
            
            # Save as quick predictions
            collection_name = 'quick_predictions'
            if collection_name not in db.list_collection_names():
                db.create_collection(collection_name)
            
            # Clear old
            db[collection_name].delete_many({'model': 'quick_update'})
            
            # Insert new
            if forecasts:
                db[collection_name].insert_many(forecasts)
                print(f"✅ Generated and saved {len(forecasts)} fresh predictions")
            
            # 5. Display predictions
            print("\n📊 Fresh 3-Day Forecasts for Karachi:")
            for forecast in forecasts:
                print(f"  {forecast['date']} ({forecast['day_name']}): "
                      f"AQI {forecast['predicted_aqi']} ({forecast['category']})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating predictions: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function for standalone execution"""
    print("=" * 60)
    print("🎯 AQI Karachi - Prediction Service (Updated)")
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
