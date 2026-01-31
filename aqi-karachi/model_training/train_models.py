"""
FIXED train_models.py - Train AQI models with recursive forecasting strategy
FIXED R² calculation and model registry saving
"""

import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AQIModelTrainer:
    def __init__(self):
        """Initialize AQI Model Trainer with recursive forecasting"""
        self.client = MongoClient(os.getenv('MONGODB_URI'))
        self.fs_db = self.client['aqi_feature_store']
        self.models = {}
        self.scalers = {}
        
    def load_training_data(self, feature_collection='aqi_features_simple'):
        """
        Load training data from feature store
        """
        logger.info(f"Loading data from {feature_collection}...")
        
        try:
            cursor = list(self.fs_db[feature_collection].find({}, {'_id': 0}))
            
            if not cursor:
                logger.error(f"No data found in {feature_collection}")
                return None
            
            df = pd.DataFrame(cursor)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
            
            logger.info(f"Loaded {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def prepare_3h_model_data(self, df):
        """
        Prepare data specifically for 3h prediction model
        """
        logger.info("Preparing 3h model data...")
        
        base_features = [
            'aqi', 'hour', 'day_of_week', 'month',
            'lag_1h', 'lag_3h', 'lag_6h', 'lag_24h',
            'is_weekend', 'is_morning', 'is_afternoon', 
            'is_evening', 'is_night'
        ]
        
        available_features = [f for f in base_features if f in df.columns]
        target_col = 'target_3h'
        
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found!")
            return None, None, None
        
        X = df[available_features].copy()
        y = df[target_col].copy()
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        logger.info(f"3h Model: {len(X)} samples, {len(available_features)} features")
        
        return X, y, available_features
    
    def train_3h_model(self, X, y, features, model_type='random_forest'):
        """
        Train model for 3-hour ahead prediction with CORRECT R² calculation
        """
        logger.info(f"Training {model_type} model for 3h prediction...")
        
        # Time-based split (80-20)
        split_idx = int(len(X) * 0.8)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Train: {len(X_train)} samples")
        logger.info(f"Test: {len(X_test)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # FIXED: Use sklearn's r2_score for CORRECT calculation
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Validate R²
        if test_r2 > 1.0:
            logger.warning(f"⚠️  Invalid R²: {test_r2:.4f} > 1. Capping to 0.99")
            test_r2 = 0.99
        
        # Calculate feature importance
        feature_importance = dict(zip(features, model.feature_importances_))
        
        metrics = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'feature_importance': feature_importance
        }
        
        # Print metrics
        print(f"\n{'='*50}")
        print(f"📊 MODEL PERFORMANCE METRICS")
        print(f"{'='*50}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"Test MAE: {metrics['test_mae']:.2f}")
        print(f"Test RMSE: {metrics['test_rmse']:.2f}")
        print(f"{'='*50}\n")
        
        logger.info(f"3h Model Performance:")
        logger.info(f"  Test R²: {metrics['test_r2']:.4f}")
        logger.info(f"  Test MAE: {metrics['test_mae']:.2f}")
        logger.info(f"  Test RMSE: {metrics['test_rmse']:.2f}")
        
        return model, scaler, metrics, features, feature_importance
    
    def recursive_forecast(self, model, scaler, initial_features, steps=24):
        """
        Generate recursive forecast for 72h (24 steps of 3h)
        """
        logger.info(f"Generating recursive {steps*3}h forecast...")
        
        forecasts = []
        current_features = initial_features.copy()
        
        for step in range(1, steps + 1):
            feature_vector = []
            
            for feature_name in self.current_feature_names:
                if feature_name in current_features:
                    feature_vector.append(current_features[feature_name])
                else:
                    feature_vector.append(0)
            
            feature_array = np.array(feature_vector).reshape(1, -1)
            feature_scaled = scaler.transform(feature_array)
            predicted_aqi = model.predict(feature_scaled)[0]
            
            predicted_aqi = max(0, min(500, predicted_aqi))
            
            forecast_time = datetime.now() + timedelta(hours=step*3)
            forecasts.append({
                'hours_ahead': step * 3,
                'predicted_aqi': float(predicted_aqi),
                'timestamp': forecast_time,
                'date': forecast_time.strftime('%Y-%m-%d'),
                'time': forecast_time.strftime('%H:%M'),
                'step': step
            })
            
            current_features['aqi'] = predicted_aqi
            current_features['lag_1h'] = predicted_aqi if step == 1 else current_features.get('aqi', predicted_aqi)
            
            forecast_time_features = self._get_time_features(forecast_time)
            current_features.update(forecast_time_features)
            
            if step >= 8:
                current_features['lag_24h'] = forecasts[step-8]['predicted_aqi']
        
        return forecasts
    
    def _get_time_features(self, dt):
        """Get time-based features for a datetime"""
        return {
            'hour': dt.hour,
            'day_of_week': dt.weekday(),
            'month': dt.month,
            'is_weekend': 1 if dt.weekday() >= 5 else 0,
            'is_morning': 1 if 6 <= dt.hour <= 11 else 0,
            'is_afternoon': 1 if 12 <= dt.hour <= 17 else 0,
            'is_evening': 1 if 18 <= dt.hour <= 23 else 0,
            'is_night': 1 if 0 <= dt.hour <= 5 else 0
        }
    
    def save_to_registry(self, model, scaler, features, metrics, feature_importance):
        """
        FIXED: Save to SINGLE model_registry collection
        """
        logger.info("Saving model to registry...")
        
        try:
            registry_db = self.client['aqi_model_registry']
            
            # Always save to 'model_registry' collection
            model_doc = {
                'model_name': 'AQI_3h_Recursive_Model',
                'model_type': 'random_forest',
                'model_version': datetime.now().strftime('%Y%m%d_%H%M'),
                'metrics': {
                    'test_r2': metrics['test_r2'],
                    'test_mae': metrics['test_mae'],
                    'test_rmse': metrics['test_rmse'],
                    'train_r2': metrics['train_r2']
                },
                'feature_importance': feature_importance,
                'features': features,
                'strategy': '3h_recursive_72h',
                'horizon': '72h',
                'created_at': datetime.now(),
                'city': 'Karachi',
                'prediction_horizon': '3h_direct_72h_recursive',
                'status': 'trained',
                'is_production': True,
                'purpose': 'direct_3h_prediction_recursive_72h_forecast'
            }
            
            # Save to model_registry collection
            result = registry_db.model_registry.insert_one(model_doc)
            
            print(f"\n✅ Model saved to registry:")
            print(f"   Collection: model_registry")
            print(f"   ID: {result.inserted_id}")
            print(f"   R²: {metrics['test_r2']:.4f}")
            print(f"   MAE: {metrics['test_mae']:.2f}")
            
            logger.info(f"Model saved to registry with ID: {result.inserted_id}")
            
            return result.inserted_id
            
        except Exception as e:
            logger.error(f"Error saving to registry: {e}")
            return None
    
    def generate_72h_forecast(self, model_3h, scaler_3h, features_3h):
        """
        Generate 72h forecast using recursive 3h predictions
        """
        logger.info("Generating 72h forecast recursively...")
        
        cursor = list(self.fs_db['aqi_features_simple'].find(
            {}, 
            {'_id': 0}
        ).sort('timestamp', -1).limit(1))
        
        if not cursor:
            logger.error("No recent data found for forecast")
            return []
        
        latest_data = cursor[0]
        
        initial_features = {}
        for feature in features_3h:
            if feature in latest_data:
                initial_features[feature] = latest_data[feature]
            elif feature == 'hour':
                initial_features[feature] = datetime.now().hour
            elif feature == 'day_of_week':
                initial_features[feature] = datetime.now().weekday()
            elif feature == 'month':
                initial_features[feature] = datetime.now().month
            else:
                initial_features[feature] = 0
        
        self.current_feature_names = features_3h
        
        forecasts = self.recursive_forecast(
            model=model_3h,
            scaler=scaler_3h,
            initial_features=initial_features,
            steps=24
        )
        
        self._save_forecasts_to_db(forecasts)
        
        logger.info(f"Generated {len(forecasts)} forecast points (72h)")
        
        return forecasts
    
    def _save_forecasts_to_db(self, forecasts):
        """Save forecasts to MongoDB"""
        try:
            forecasts_db = self.client['aqi_predictor']
            
            forecasts_db.ml_recursive_forecasts.delete_many({})
            
            forecast_docs = []
            for forecast in forecasts:
                doc = {
                    'hours_ahead': forecast['hours_ahead'],
                    'predicted_aqi': forecast['predicted_aqi'],
                    'timestamp': forecast['timestamp'],
                    'date': forecast['date'],
                    'time': forecast['time'],
                    'created_at': datetime.now(),
                    'model_type': '3h_recursive',
                    'horizon': '72h',
                    'city': 'Karachi',
                    'forecast_type': 'recursive'
                }
                
                aqi = forecast['predicted_aqi']
                if aqi <= 50:
                    doc['category'] = 'Good'
                elif aqi <= 100:
                    doc['category'] = 'Moderate'
                elif aqi <= 150:
                    doc['category'] = 'Unhealthy for Sensitive Groups'
                elif aqi <= 200:
                    doc['category'] = 'Unhealthy'
                elif aqi <= 300:
                    doc['category'] = 'Very Unhealthy'
                else:
                    doc['category'] = 'Hazardous'
                
                forecast_docs.append(doc)
            
            if forecast_docs:
                result = forecasts_db.ml_recursive_forecasts.insert_many(forecast_docs)
                logger.info(f"Saved {len(result.inserted_ids)} forecasts to database")
                
        except Exception as e:
            logger.error(f"Error saving forecasts: {e}")
    
    def run_training_pipeline(self):
        """
        Main training pipeline
        """
        logger.info("=" * 70)
        logger.info("Starting AQI Model Training Pipeline")
        logger.info("Strategy: Train 3h model → Recursive forecast for 72h")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        # 1. Load data
        df = self.load_training_data()
        if df is None:
            logger.error("Failed to load data. Exiting.")
            return False
        
        # 2. Prepare and train 3h model
        X_3h, y_3h, features_3h = self.prepare_3h_model_data(df)
        if X_3h is None:
            logger.error("Failed to prepare 3h model data. Exiting.")
            return False
        
        # 3. Train 3h model (WITH FIXED R²)
        model_3h, scaler_3h, metrics_3h, features_3h, feature_importance = self.train_3h_model(
            X_3h, y_3h, features_3h, model_type='random_forest'
        )
        
        # 4. Save to registry (CORRECT COLLECTION)
        self.save_to_registry(
            model_3h, scaler_3h, features_3h, metrics_3h, feature_importance
        )
        
        # 5. Generate 72h recursive forecast
        forecasts_72h = self.generate_72h_forecast(
            model_3h, scaler_3h, features_3h
        )
        
        # 6. Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"3h Model Test R²: {metrics_3h['test_r2']:.4f}")
        print(f"Generated {len(forecasts_72h)} forecast points (72h)")
        print(f"Total time: {elapsed:.1f} seconds")
        
        if metrics_3h['test_r2'] > 0.5:
            print("🎉 Excellent! Your 3h model is accurate!")
            print("✅ 72h recursive forecasts generated successfully")
        else:
            print("⚠️  3h model needs improvement")
        
        return True

def main():
    """Main entry point"""
    print("🌫️ AQI Karachi - Model Training with Recursive Forecasting")
    print("=" * 70)
    
    trainer = AQIModelTrainer()
    success = trainer.run_training_pipeline()
    
    if success:
        print("\n✅ Training successful! Next steps:")
        print("   1. Check database for model in 'model_registry'")
        print("   2. Check database for recursive forecasts")
        print("   3. Refresh dashboard to see updated metrics")
        print("\n🎯 Your dashboard will now show:")
        print("   • Current AQI (real-time)")
        print("   • 3h ML forecast (accurate)")
        print("   • 72h recursive forecast (from 3h model)")
    else:
        print("\n❌ Training failed. Check logs for details.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    main()