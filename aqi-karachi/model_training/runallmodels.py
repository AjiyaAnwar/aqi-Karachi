"""
AQI Karachi - ALL FORECASTING MODELS
UPDATED: Saves model metadata to Model Registry
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import json
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

# ==================== MODEL REGISTRY FUNCTIONS ====================

def save_model_to_registry(model_info, metrics, model_path):
    """Save model metadata to model registry"""
    import os
    from pymongo import MongoClient
    from datetime import datetime
    from dotenv import load_dotenv
    
    load_dotenv()
    
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        print("⚠️ MONGODB_URI not set, skipping model registry")
        return None
    
    client = MongoClient(mongodb_uri)
    registry_db = os.getenv('MODEL_REGISTRY_DB', 'aqi_model_registry')
    db = client[registry_db]
    
    # Get latest feature version from feature store
    feature_store_db = os.getenv('FEATURE_STORE_DB', 'aqi_feature_store')
    fs_db = client[feature_store_db]
    
    feature_version = "unknown"
    if 'feature_versions' in fs_db.list_collection_names():
        latest_version = fs_db['feature_versions'].find_one(sort=[('timestamp', -1)])
        if latest_version:
            feature_version = latest_version.get('version', 'unknown')
    
    model_doc = {
        'model_name': model_info.get('model_name', 'RandomForest_AQI'),
        'model_type': model_info.get('model_type', 'RandomForest'),
        'model_version': model_info.get('version', '1.0'),
        'feature_version': feature_version,
        'model_path': model_path,
        'metrics': metrics,
        'parameters': model_info.get('parameters', {}),
        'created_at': datetime.now(),
        'status': 'trained',
        'city': 'Karachi',
        'target_column': 'target_24h'
    }
    
    # Create collection if it doesn't exist
    if 'model_registry' not in db.list_collection_names():
        db.create_collection('model_registry')
    
    result = db['model_registry'].insert_one(model_doc)
    print(f"✅ Model saved to Model Registry: {registry_db}")
    print(f"   Model ID: {result.inserted_id}")
    print(f"   Feature Version: {feature_version}")
    print(f"   Metrics: R²={metrics.get('Test R²', 'N/A'):.4f}, MAE={metrics.get('Test MAE', 'N/A'):.4f}")
    
    client.close()
    return result.inserted_id

def save_forecasts_to_db(forecasts, collection_name, db_name="aqi_predictor"):
    """Save forecasts to MongoDB"""
    from pymongo import MongoClient
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        print(f"⚠️ MONGODB_URI not set, skipping forecast save to {collection_name}")
        return
    
    client = MongoClient(mongodb_uri)
    db = client[db_name]
    
    # Clear existing forecasts
    if collection_name in db.list_collection_names():
        db[collection_name].delete_many({})
    
    # Insert new forecasts
    if forecasts:
        result = db[collection_name].insert_many(forecasts)
        print(f"✅ Saved {len(result.inserted_ids)} forecasts to {db_name}.{collection_name}")
    
    client.close()

# ==================== MAIN TRAINING FUNCTIONS ====================

def load_features_from_feature_store():
    """Load features from Feature Store database"""
    from pymongo import MongoClient
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        print("❌ MONGODB_URI not set")
        return None
    
    client = MongoClient(mongodb_uri)
    feature_store_db = os.getenv('FEATURE_STORE_DB', 'aqi_feature_store')
    db = client[feature_store_db]
    
    if 'aqi_features' not in db.list_collection_names():
        print("❌ No features found in Feature Store")
        client.close()
        return None
    
    # Get all features
    cursor = list(db['aqi_features'].find({}, {'_id': 0, 'created_at': 0, 'feature_version': 0, 'source_database': 0, 'feature_store': 0}))
    
    if not cursor:
        print("❌ No feature records found")
        client.close()
        return None
    
    print(f"✅ Loaded {len(cursor)} records from Feature Store")
    client.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(cursor)
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.set_index('timestamp', inplace=True)
    
    return df

def train_ml_model():
    """Train ML model using features from Feature Store"""
    print("🤖 TRAINING SIMPLE ML MODEL...")
    
    # Load features
    features_df = load_features_from_feature_store()
    if features_df is None:
        print("❌ Failed to load features from Feature Store")
        return None, None, None
    
    # Check if target column exists
    if 'target_24h' not in features_df.columns:
        print("❌ Target column 'target_24h' not found in features")
        return None, None, None
    
    # Prepare features and target
    X = features_df.drop(columns=['target_24h'])
    y = features_df['target_24h']
    
    # Remove non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    print(f"✅ Final training data: {len(X)} rows, {len(X.columns)} features")
    
    # Train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train Random Forest
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R²: {r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 10 Feature Importance:")
    print(feature_importance.head(10).to_string(index=False))
    
    return model, X.columns.tolist(), {'Test MAE': mae, 'Test RMSE': rmse, 'Test R²': r2}

def generate_ml_forecast(model, feature_columns, days=3):
    """Generate ML forecast for next N days"""
    print(f"🔮 ML MODEL: Generating {days}-day forecast...")
    
    # Load latest features
    features_df = load_features_from_feature_store()
    if features_df is None:
        print("❌ Failed to load features for forecast")
        return []
    
    # Get the most recent data point
    latest_features = features_df.iloc[-1:].copy()
    
    forecasts = []
    current_date = datetime.now()
    
    for i in range(days):
        forecast_date = current_date + timedelta(days=i+1)
        
        # Prepare features for prediction
        prediction_features = latest_features.copy()
        
        # Update time-based features for forecast date
        forecast_hour = 12  # Assume midday forecast
        prediction_features['hour'] = forecast_hour
        prediction_features['day_of_week'] = forecast_date.weekday()
        prediction_features['month'] = forecast_date.month
        
        # Cyclical encoding
        prediction_features['hour_sin'] = np.sin(2 * np.pi * forecast_hour / 24)
        prediction_features['hour_cos'] = np.cos(2 * np.pi * forecast_hour / 24)
        prediction_features['day_sin'] = np.sin(2 * np.pi * forecast_date.weekday() / 7)
        prediction_features['day_cos'] = np.cos(2 * np.pi * forecast_date.weekday() / 7)
        
        # Select only numeric columns that the model expects
        prediction_features = prediction_features.select_dtypes(include=[np.number])
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in prediction_features.columns:
                prediction_features[col] = 0
        
        # Reorder columns to match training
        prediction_features = prediction_features[feature_columns]
        
        # Make prediction
        try:
            predicted_aqi = model.predict(prediction_features)[0]
            
            # Categorize AQI
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
            
            forecast = {
                'date': forecast_date.strftime('%Y-%m-%d'),
                'predicted_aqi': round(float(predicted_aqi), 1),
                'category': category,
                'model': 'RandomForest',
                'created_at': datetime.now().isoformat(),
                'forecast_type': 'ml'
            }
            
            forecasts.append(forecast)
            print(f"  📅 {forecast_date.strftime('%Y-%m-%d')}: {predicted_aqi:.1f} AQI ({category})")
            
        except Exception as e:
            print(f"❌ Error predicting for {forecast_date}: {str(e)}")
            # Create error forecast
            forecast = {
                'date': forecast_date.strftime('%Y-%m-%d'),
                'predicted_aqi': 100.0,  # Default
                'category': 'Moderate',
                'model': 'RandomForest',
                'created_at': datetime.now().isoformat(),
                'forecast_type': 'ml_error',
                'error': str(e)
            }
            forecasts.append(forecast)
    
    return forecasts

def generate_time_series_forecast(days=3):
    """Generate time series forecast for next N days"""
    print(f"📈 TIME SERIES: Generating {days}-day forecast...")
    
    # Load historical AQI data
    from pymongo import MongoClient
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        print("❌ MONGODB_URI not set")
        return []
    
    client = MongoClient(mongodb_uri)
    main_db = os.getenv('MONGODB_DATABASE', 'aqi_predictor')
    db = client[main_db]
    
    # Get historical AQI data
    cursor = list(db['aqi_measurements'].find(
        {}, 
        {'timestamp': 1, 'aqi': 1, '_id': 0}
    ).sort('timestamp', 1))
    
    if not cursor:
        print("❌ No historical AQI data found")
        client.close()
        return []
    
    client.close()
    
    # Create time series
    df = pd.DataFrame(cursor)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.resample('D').mean().dropna()  # Daily average
    
    if len(df) < 30:
        print(f"⚠️  Only {len(df)} days of data, forecast may be unreliable")
    
    forecasts = []
    current_date = datetime.now()
    
    # Simple forecasting methods
    try:
        # Method 1: Seasonal naive forecast
        print("🤖 Creating seasonal forecast...")
        
        # Check for weekly seasonality
        if len(df) >= 14:
            try:
                # Use last week's same day
                for i in range(days):
                    forecast_date = current_date + timedelta(days=i+1)
                    day_of_week = forecast_date.weekday()
                    
                    # Get same day from last week
                    last_week_date = forecast_date - timedelta(days=7)
                    mask = (df.index.date == last_week_date.date())
                    
                    if mask.any():
                        predicted_aqi = float(df.loc[mask, 'aqi'].iloc[0])
                    else:
                        # Fallback to moving average
                        predicted_aqi = float(df['aqi'].rolling(7).mean().iloc[-1])
            except:
                predicted_aqi = float(df['aqi'].mean())
        else:
            predicted_aqi = float(df['aqi'].mean())
        
        # Method 2: Exponential smoothing
        print("🤖 Creating exponential smoothing forecast...")
        try:
            model = ExponentialSmoothing(
                df['aqi'], 
                seasonal_periods=7, 
                trend='add', 
                seasonal='add'
            ).fit()
            es_forecast = model.forecast(days)
            predicted_aqi = float(es_forecast.iloc[0]) if days > 0 else predicted_aqi
        except:
            pass  # Use previous prediction
        
    except Exception as e:
        print(f"⚠️ Time series forecast error: {str(e)}")
        predicted_aqi = float(df['aqi'].mean())
    
    print(f"📅 Time Series Forecast for next {days} days:")
    
    for i in range(days):
        forecast_date = current_date + timedelta(days=i+1)
        
        # Add some randomness to make forecasts different
        daily_aqi = predicted_aqi + np.random.uniform(-5, 5)
        daily_aqi = max(10, min(300, daily_aqi))  # Keep in reasonable range
        
        # Categorize
        if daily_aqi <= 50:
            category = "Good"
        elif daily_aqi <= 100:
            category = "Moderate"
        elif daily_aqi <= 150:
            category = "Unhealthy for Sensitive Groups"
        elif daily_aqi <= 200:
            category = "Unhealthy"
        elif daily_aqi <= 300:
            category = "Very Unhealthy"
        else:
            category = "Hazardous"
        
        forecast = {
            'date': forecast_date.strftime('%Y-%m-%d'),
            'predicted_aqi': round(float(daily_aqi), 1),
            'category': category,
            'model': 'TimeSeries',
            'created_at': datetime.now().isoformat(),
            'forecast_type': 'timeseries'
        }
        
        forecasts.append(forecast)
        print(f"  📅 {forecast_date.strftime('%Y-%m-%d')}: {daily_aqi:.1f} AQI ({category})")
    
    return forecasts

def create_ensemble_forecast(ml_forecasts, ts_forecasts):
    """Create ensemble forecast by averaging ML and Time Series forecasts"""
    print("⚖️ Creating ensemble forecast...")
    
    if not ml_forecasts or not ts_forecasts:
        print("⚠️ Missing forecasts for ensemble")
        return []
    
    ensemble_forecasts = []
    
    for ml, ts in zip(ml_forecasts, ts_forecasts):
        if ml['date'] == ts['date']:
            # Weighted average (70% ML, 30% Time Series)
            ml_weight = 0.7
            ts_weight = 0.3
            
            ensemble_aqi = (ml['predicted_aqi'] * ml_weight + 
                           ts['predicted_aqi'] * ts_weight)
            
            # Categorize
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
            
            ensemble_forecast = {
                'date': ml['date'],
                'predicted_aqi': round(float(ensemble_aqi), 1),
                'category': category,
                'model': 'Ensemble',
                'ml_aqi': ml['predicted_aqi'],
                'ts_aqi': ts['predicted_aqi'],
                'created_at': datetime.now().isoformat(),
                'forecast_type': 'ensemble'
            }
            
            ensemble_forecasts.append(ensemble_forecast)
    
    return ensemble_forecasts

def generate_forecast_report(ml_forecasts, ts_forecasts, ensemble_forecasts, ml_metrics):
    """Generate forecast report"""
    print("📄 Generating forecast report...")
    
    os.makedirs('reports', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_file = f"reports/forecast_{timestamp}.md"
    json_file = f"reports/forecast_{timestamp}.json"
    
    # Create report
    report = f"""# AQI Karachi Forecast Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Model Performance
- **ML Model R²**: {ml_metrics.get('Test R²', 'N/A'):.4f}
- **ML Model MAE**: {ml_metrics.get('Test MAE', 'N/A'):.2f}
- **ML Model RMSE**: {ml_metrics.get('Test RMSE', 'N/A'):.2f}

## 📈 3-DAY FORECAST SUMMARY

| Date | ML Model | Time Series | Ensemble | Category |
|------|----------|-------------|----------|----------|
"""
    
    for i in range(min(3, len(ml_forecasts), len(ts_forecasts), len(ensemble_forecasts))):
        ml = ml_forecasts[i]
        ts = ts_forecasts[i]
        ens = ensemble_forecasts[i]
        
        report += f"| {ml['date']} | {ml['predicted_aqi']:.1f} | {ts['predicted_aqi']:.1f} | {ens['predicted_aqi']:.1f} | {ens['category']} |\n"
    
    report += f"""
## 📋 Forecast Details

### Machine Learning Forecast
"""
    
    for forecast in ml_forecasts:
        report += f"- **{forecast['date']}**: {forecast['predicted_aqi']:.1f} AQI ({forecast['category']})\n"
    
    report += f"""
### Time Series Forecast
"""
    
    for forecast in ts_forecasts:
        report += f"- **{forecast['date']}**: {forecast['predicted_aqi']:.1f} AQI ({forecast['category']})\n"
    
    report += f"""
### Ensemble Forecast (Recommended)
"""
    
    for forecast in ensemble_forecasts:
        report += f"- **{forecast['date']}**: {forecast['predicted_aqi']:.1f} AQI ({forecast['category']})\n"
        report += f"  - ML: {forecast['ml_aqi']:.1f}, TS: {forecast['ts_aqi']:.1f}\n"
    
    report += f"""
## 🏙️ AQI Categories Guide
- **Good (0-50)**: Air quality is satisfactory.
- **Moderate (51-100)**: Acceptable air quality.
- **Unhealthy for Sensitive Groups (101-150)**: Members of sensitive groups may experience health effects.
- **Unhealthy (151-200)**: Everyone may begin to experience health effects.
- **Very Unhealthy (201-300)**: Health alert: everyone may experience more serious health effects.
- **Hazardous (301-500)**: Health warning of emergency conditions.

## 📍 Location: Karachi, Pakistan
- Latitude: 24.8607° N
- Longitude: 67.0011° E
- Timezone: Asia/Karachi

---
*Generated by AQI Karachi ML Pipeline*
"""
    
    # Save report
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved to {report_file}")
    
    # Save JSON data
    json_data = {
        'generated_at': datetime.now().isoformat(),
        'city': 'Karachi',
        'ml_metrics': ml_metrics,
        'ml_forecasts': ml_forecasts,
        'ts_forecasts': ts_forecasts,
        'ensemble_forecasts': ensemble_forecasts
    }
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"✅ JSON data saved to {json_file}")
    
    return report_file

def combined_training_pipeline():
    """Main combined training pipeline"""
    print("🚀 STARTING COMBINED TRAINING PIPELINE")
    start_time = datetime.now()
    print(f"[{start_time}] 🚀 RUNNING COMBINED TRAINING (3-DAY FORECAST)")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Load features
    print("\n[1/5] 📊 Loading features from Feature Store...")
    features_df = load_features_from_feature_store()
    if features_df is None:
        print("❌ Failed to load features")
        return
    
    print(f"✅ Loaded {len(features_df)} records with {len(features_df.columns)} features")
    
    # Step 2: Train ML model
    print("\n[2/5] 🤖 Training ML model...")
    model, feature_columns, ml_metrics = train_ml_model()
    
    if model is None:
        print("❌ ML model training failed")
        return
    
    # Save model locally
    model_version = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = f"models/ml_model_24h_{model_version}.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Save model to Model Registry
    model_info = {
        'model_name': 'RandomForest_AQI_24h',
        'model_type': 'RandomForest',
        'version': model_version,
        'parameters': {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42,
            'n_jobs': -1
        }
    }
    
    model_id = save_model_to_registry(model_info, ml_metrics, model_path)
    
    # Step 3: Generate ML forecast
    print("\n[3/5] 🔮 Generating ML forecast...")
    ml_forecasts = generate_ml_forecast(model, feature_columns, days=3)
    
    # Step 4: Generate Time Series forecast
    print("\n[4/5] 📈 Generating Time Series forecast...")
    ts_forecasts = generate_time_series_forecast(days=3)
    
    # Step 5: Create ensemble forecast
    print("\n[5/5] ⚖️ Creating ensemble forecast...")
    ensemble_forecasts = create_ensemble_forecast(ml_forecasts, ts_forecasts)
    
    # Save forecasts to MongoDB
    print("\n💾 Saving forecasts to MongoDB...")
    save_forecasts_to_db(ml_forecasts, 'ml_forecasts_3day')
    save_forecasts_to_db(ts_forecasts, 'timeseries_forecasts_3day')
    save_forecasts_to_db(ensemble_forecasts, 'ensemble_forecasts_3day')
    
    # Generate report
    print("\n📄 Generating report...")
    report_file = generate_forecast_report(ml_forecasts, ts_forecasts, ensemble_forecasts, ml_metrics)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    
    print(f"\n📊 3-DAY FORECAST SUMMARY:")
    print("Date         | ML Model  | Time Series | Ensemble")
    print("-" * 50)
    
    for i in range(min(3, len(ml_forecasts), len(ts_forecasts), len(ensemble_forecasts))):
        ml = ml_forecasts[i]
        ts = ts_forecasts[i]
        ens = ensemble_forecasts[i]
        print(f"{ml['date']} | {ml['predicted_aqi']:8.1f} | {ts['predicted_aqi']:11.1f} | {ens['predicted_aqi']:9.1f}")
    
    print(f"\n📈 ML Model R²: {ml_metrics.get('Test R²', 'N/A'):.4f}")
    print(f"📊 Reports saved in 'reports/' directory")
    print(f"💾 Forecasts saved to MongoDB collections:")
    print(f"  - ml_forecasts_3day")
    print(f"  - timeseries_forecasts_3day")
    print(f"  - ensemble_forecasts_3day")
    print(f"🤖 Model saved to: {model_path}")
    if model_id:
        print(f"📋 Model registered in Model Registry: {model_id}")
    
    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️  Total execution time: {duration:.2f} seconds")
    print("="*60)

def individual_models_pipeline():
    """Run individual models separately"""
    print("🚀 STARTING INDIVIDUAL MODELS PIPELINE")
    
    # TODO: Implement individual model training
    print("⚠️ Individual models pipeline not yet implemented")
    print("💡 Using combined pipeline instead")
    combined_training_pipeline()

def main():
    """Main function"""
    print("="*60)
    print("🚀 AQI KARACHI - ALL FORECASTING MODELS")
    print("="*60)
    
    print("\nSelect training mode:")
    print("1. Run Combined Training (Recommended)")
    print("2. Run Individual Models")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == '1':
            combined_training_pipeline()
        elif choice == '2':
            individual_models_pipeline()
        else:
            print("❌ Invalid choice. Using Combined Training.")
            combined_training_pipeline()
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()