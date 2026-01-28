"""
Combined Model Training Pipeline for AQI Forecasting
Trains both ML and Time Series models, generates 3-day forecasts
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# MongoDB
from pymongo import MongoClient

load_dotenv()

def get_combined_historical_data():
    """Get historical AQI data for both models"""
    print("📥 Loading historical data...")
    
    try:
        client = MongoClient(os.getenv('MONGODB_URI'))
        db = client[os.getenv('MONGODB_DATABASE')]
        
        # Get data - exclude ObjectId
        cursor = db.aqi_features.find({}, {'_id': 0})
        df = pd.DataFrame(list(cursor))
        client.close()
        
        if df.empty:
            print("❌ No data found!")
            return None
        
        # Process data
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Ensure AQI column exists
        if 'aqi' not in df.columns:
            if 'us_aqi' in df.columns:
                df['aqi'] = df['us_aqi']
            else:
                print("❌ No AQI column found!")
                return None
        
        # Clean data
        df = df[(df['aqi'] >= 0) & (df['aqi'] <= 500)]
        df = df.drop_duplicates()
        
        print(f"✅ Loaded {len(df)} records")
        print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"📊 AQI stats - Min: {df['aqi'].min():.1f}, Max: {df['aqi'].max():.1f}, Mean: {df['aqi'].mean():.1f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def train_ml_model_simple(df):
    """
    Simple ML model training that works with your existing data
    """
    print("\n🤖 TRAINING SIMPLE ML MODEL...")
    
    try:
        # Make copy for feature engineering
        df_ml = df.copy()
        
        # Create time features
        df_ml['hour'] = df_ml['timestamp'].dt.hour
        df_ml['day_of_week'] = df_ml['timestamp'].dt.dayofweek
        
        # Create target - AQI 24 hours from now
        df_ml['target'] = df_ml['aqi'].shift(-24)
        df_ml = df_ml.dropna(subset=['target'])
        
        if len(df_ml) < 100:
            print(f"❌ Not enough data after target creation: {len(df_ml)} rows")
            return None, None, None, 0
        
        # Create simple features
        df_ml['lag_1h'] = df_ml['aqi'].shift(1)
        df_ml['lag_24h'] = df_ml['aqi'].shift(24)
        df_ml['rolling_mean_6h'] = df_ml['aqi'].rolling(window=6, min_periods=1).mean()
        
        # Time features
        df_ml['hour_sin'] = np.sin(2 * np.pi * df_ml['hour'] / 24)
        df_ml['hour_cos'] = np.cos(2 * np.pi * df_ml['hour'] / 24)
        df_ml['is_weekend'] = (df_ml['day_of_week'] >= 5).astype(int)
        
        # Remove NaN
        df_ml = df_ml.dropna()
        
        if len(df_ml) < 50:
            print(f"❌ Not enough data after cleaning: {len(df_ml)} rows")
            return None, None, None, 0
        
        print(f"✅ Final training data: {len(df_ml)} rows")
        
        # Prepare features and target
        exclude_cols = ['timestamp', 'aqi', 'target']
        feature_cols = [col for col in df_ml.columns if col not in exclude_cols]
        
        print(f"📋 Using {len(feature_cols)} features: {feature_cols}")
        
        # Ensure all features are numeric
        for col in feature_cols:
            df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')
        
        # Fill NaN with column mean
        df_ml[feature_cols] = df_ml[feature_cols].fillna(df_ml[feature_cols].mean())
        
        X = df_ml[feature_cols].values
        y = df_ml['target'].values
        
        # Split data (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train a simple model
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_absolute_error
        
        print("Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"✅ Model trained successfully!")
        print(f"   Train R²: {train_r2:.4f}")
        print(f"   Test R²: {test_r2:.4f}")
        print(f"   Test MAE: {test_mae:.2f}")
        
        # Save model
        import joblib
        os.makedirs('models', exist_ok=True)
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_cols,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'trained_at': datetime.now(),
            'n_samples': len(X_train)
        }
        
        model_path = f'models/ml_model_24h_{datetime.now().strftime("%Y%m%d_%H%M")}.joblib'
        joblib.dump(model_data, model_path)
        print(f"✅ Model saved to {model_path}")
        
        return model, scaler, feature_cols, test_r2
        
    except Exception as e:
        print(f"❌ Error training ML model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, 0

def generate_3day_forecast_ml(model, scaler, feature_cols, df):
    """
    Generate 3-day forecast using ML model
    """
    print("\n🔮 ML MODEL: Generating 3-day forecast...")
    
    # If no model, use simple fallback
    if model is None or scaler is None:
        print("⚠️ Using simple forecast fallback (no trained model)")
        return generate_simple_fallback_forecast(df, model_type='ml_fallback')
    
    try:
        last_time = df['timestamp'].iloc[-1]
        forecasts = []
        
        for day_offset in range(1, 4):
            forecast_date = last_time.date() + timedelta(days=day_offset)
            
            # Predict for midday (12:00) as representative
            forecast_time = datetime.combine(forecast_date, datetime.min.time()) + timedelta(hours=12)
            
            # Create features for prediction
            features = {}
            
            # Time features
            features['hour'] = 12  # Midday
            features['day_of_week'] = forecast_time.weekday()
            
            # Cyclical encoding
            features['hour_sin'] = np.sin(2 * np.pi * 12 / 24)
            features['hour_cos'] = np.cos(2 * np.pi * 12 / 24)
            features['is_weekend'] = 1 if forecast_time.weekday() >= 5 else 0
            
            # Lag features - use recent averages
            recent_avg = df['aqi'].tail(24).mean()
            features['lag_1h'] = recent_avg
            features['lag_24h'] = df['aqi'].tail(24).mean()
            features['rolling_mean_6h'] = df['aqi'].tail(6).mean()
            
            # Fill missing features
            for col in feature_cols:
                if col not in features:
                    if col in df.columns:
                        features[col] = df[col].mean()
                    else:
                        features[col] = 0
            
            # Create feature array
            feature_array = np.array([[features.get(col, 0) for col in feature_cols]])
            
            try:
                # Scale and predict
                feature_scaled = scaler.transform(feature_array)
                predicted_aqi = float(model.predict(feature_scaled)[0])
                predicted_aqi = max(0, min(500, predicted_aqi))
            except:
                # If prediction fails, use historical average
                predicted_aqi = df['aqi'].mean()
            
            # Determine category
            if predicted_aqi <= 50:
                category = "Good"
            elif predicted_aqi <= 100:
                category = "Moderate"
            elif predicted_aqi <= 150:
                category = "Unhealthy for Sensitive Groups"
            elif predicted_aqi <= 200:
                category = "Unhealthy"
            else:
                category = "Very Unhealthy"
            
            forecasts.append({
                'date': forecast_date.strftime('%Y-%m-%d'),
                'predicted_aqi': round(predicted_aqi, 1),
                'category': category,
                'model': 'ml'
            })
        
        forecast_df = pd.DataFrame(forecasts)
        
        print(f"📅 ML Forecast for next 3 days:")
        for _, row in forecast_df.iterrows():
            print(f"  {row['date']}: AQI {row['predicted_aqi']:.1f} ({row['category']})")
        
        return forecast_df
        
    except Exception as e:
        print(f"❌ Error generating ML forecast: {e}")
        return generate_simple_fallback_forecast(df, model_type='ml_error')

def generate_simple_fallback_forecast(df, model_type='fallback'):
    """
    Generate simple fallback forecast when ML model fails
    """
    print(f"⚠️ Using {model_type} forecast")
    
    last_time = df['timestamp'].iloc[-1]
    historical_avg = df['aqi'].mean()
    
    forecasts = []
    
    for day_offset in range(1, 4):
        forecast_date = last_time.date() + timedelta(days=day_offset)
        day_of_week = forecast_date.weekday()
        
        # Base on historical average with day-of-week adjustment
        if day_of_week < 5:  # Weekday
            predicted_aqi = historical_avg * 1.05
        else:  # Weekend
            predicted_aqi = historical_avg * 0.95
        
        # Add small variation
        predicted_aqi += np.random.normal(0, 5)
        predicted_aqi = max(60, min(120, predicted_aqi))
        
        if predicted_aqi <= 50:
            category = "Good"
        elif predicted_aqi <= 100:
            category = "Moderate"
        elif predicted_aqi <= 150:
            category = "Unhealthy for Sensitive Groups"
        elif predicted_aqi <= 200:
            category = "Unhealthy"
        else:
            category = "Very Unhealthy"
        
        forecasts.append({
            'date': forecast_date.strftime('%Y-%m-%d'),
            'predicted_aqi': round(predicted_aqi, 1),
            'category': category,
            'model': model_type
        })
    
    return pd.DataFrame(forecasts)

def generate_3day_forecast_timeseries(df):
    """
    Generate 3-day forecast using time series models
    """
    print("\n📈 TIME SERIES: Generating 3-day forecast...")
    
    try:
        # Try to import your existing time series functions
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model_training.train_time_series_models import (
            simple_seasonal_forecast, exponential_smoothing_forecast
        )
        
        # Generate forecasts
        seasonal_pred = simple_seasonal_forecast(df, periods=72)
        es_pred = exponential_smoothing_forecast(df, periods=72)
        
        # Combine and average
        all_preds = pd.concat([seasonal_pred, es_pred], ignore_index=True)
        all_preds['timestamp'] = pd.to_datetime(all_preds['timestamp'])
        all_preds['date'] = all_preds['timestamp'].dt.date
        
        daily_avg = all_preds.groupby('date').agg({
            'predicted_aqi': 'mean'
        }).reset_index().head(3)
        
        forecasts = []
        for _, row in daily_avg.iterrows():
            aqi = row['predicted_aqi']
            
            if aqi <= 50:
                category = "Good"
            elif aqi <= 100:
                category = "Moderate"
            elif aqi <= 150:
                category = "Unhealthy for Sensitive Groups"
            elif aqi <= 200:
                category = "Unhealthy"
            else:
                category = "Very Unhealthy"
            
            forecasts.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'predicted_aqi': round(aqi, 1),
                'category': category,
                'model': 'timeseries'
            })
        
        forecast_df = pd.DataFrame(forecasts)
        
        print(f"📅 Time Series Forecast for next 3 days:")
        for _, row in forecast_df.iterrows():
            print(f"  {row['date']}: AQI {row['predicted_aqi']:.1f} ({row['category']})")
        
        return forecast_df
        
    except Exception as e:
        print(f"❌ Error in time series forecast: {e}")
        return generate_simple_fallback_forecast(df, model_type='timeseries_fallback')

def save_forecasts_to_mongodb(ml_forecasts, ts_forecasts):
    """Save both forecasts to MongoDB"""
    print("\n💾 Saving forecasts to MongoDB...")
    
    try:
        client = MongoClient(os.getenv('MONGODB_URI'))
        db = client[os.getenv('MONGODB_DATABASE')]
        
        # Clear old forecasts
        db.ml_forecasts_3day.delete_many({})
        db.timeseries_forecasts_3day.delete_many({})
        db.ensemble_forecasts_3day.delete_many({})
        
        # Save ML forecasts
        ml_records = ml_forecasts.to_dict('records')
        for record in ml_records:
            record['created_at'] = datetime.now()
            record['forecast_type'] = '3_day'
        
        if ml_records:
            db.ml_forecasts_3day.insert_many(ml_records)
            print(f"✅ Saved {len(ml_records)} ML forecasts")
        
        # Save Time Series forecasts
        ts_records = ts_forecasts.to_dict('records')
        for record in ts_records:
            record['created_at'] = datetime.now()
            record['forecast_type'] = '3_day'
        
        if ts_records:
            db.timeseries_forecasts_3day.insert_many(ts_records)
            print(f"✅ Saved {len(ts_records)} Time Series forecasts")
        
        # Save ensemble forecasts (average)
        ensemble_records = []
        if len(ml_records) == len(ts_records):
            for ml_row, ts_row in zip(ml_records, ts_records):
                avg_aqi = (ml_row['predicted_aqi'] + ts_row['predicted_aqi']) / 2
                
                if avg_aqi <= 50:
                    category = "Good"
                elif avg_aqi <= 100:
                    category = "Moderate"
                elif avg_aqi <= 150:
                    category = "Unhealthy for Sensitive Groups"
                elif avg_aqi <= 200:
                    category = "Unhealthy"
                else:
                    category = "Very Unhealthy"
                
                ensemble_records.append({
                    'date': ml_row['date'],
                    'predicted_aqi': round(avg_aqi, 1),
                    'category': category,
                    'model': 'ensemble',
                    'ml_aqi': ml_row['predicted_aqi'],
                    'ts_aqi': ts_row['predicted_aqi'],
                    'created_at': datetime.now(),
                    'forecast_type': '3_day'
                })
            
            if ensemble_records:
                db.ensemble_forecasts_3day.insert_many(ensemble_records)
                print(f"✅ Saved {len(ensemble_records)} ensemble forecasts")
        
        client.close()
        print("✅ All forecasts saved to MongoDB")
        
    except Exception as e:
        print(f"❌ Error saving to MongoDB: {e}")

def generate_report(ml_forecasts, ts_forecasts):
    """Generate markdown report"""
    print("\n📄 Generating forecast report...")
    
    try:
        report = "# ML Model Forecast\n\n## Time Series Forecast\n\n"
        
        # Add time series forecasts
        for _, row in ts_forecasts.iterrows():
            report += f"### {row['date']}\n"
            report += f"**{row['predicted_aqi']:.0f}**\n\n"
            report += f"{row['category']}\n\n"
            report += "---\n\n"
        
        report += "\n## ML Model Forecast\n\n"
        
        # Add ML forecasts
        for _, row in ml_forecasts.iterrows():
            report += f"### {row['date']}\n"
            report += f"**{row['predicted_aqi']:.0f}**\n\n"
            report += f"{row['category']}\n\n"
            report += "---\n\n"
        
        # Save report
        os.makedirs('reports', exist_ok=True)
        report_file = f'reports/forecast_{datetime.now().strftime("%Y%m%d_%H%M")}.md'
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved to {report_file}")
        
        # Also save as JSON
        import json
        json_data = {
            'generated_at': datetime.now().isoformat(),
            'time_series_forecast': ts_forecasts.to_dict('records'),
            'ml_forecast': ml_forecasts.to_dict('records')
        }
        
        # Add ensemble if available
        if len(ml_forecasts) == len(ts_forecasts):
            ensemble_data = []
            for i in range(len(ml_forecasts)):
                ml_row = ml_forecasts.iloc[i]
                ts_row = ts_forecasts.iloc[i]
                avg_aqi = (ml_row['predicted_aqi'] + ts_row['predicted_aqi']) / 2
                
                if avg_aqi <= 50:
                    category = "Good"
                elif avg_aqi <= 100:
                    category = "Moderate"
                elif avg_aqi <= 150:
                    category = "Unhealthy for Sensitive Groups"
                elif avg_aqi <= 200:
                    category = "Unhealthy"
                else:
                    category = "Very Unhealthy"
                
                ensemble_data.append({
                    'date': ml_row['date'],
                    'predicted_aqi': round(avg_aqi, 1),
                    'category': category
                })
            
            json_data['ensemble_forecast'] = ensemble_data
        
        json_file = f'reports/forecast_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✅ JSON data saved to {json_file}")
        
        return report
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return ""

def main():
    """Main training pipeline"""
    print("=" * 70)
    print("🚀 AQI KARACHI - COMBINED 3-DAY FORECASTING PIPELINE")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # 1. Load data
    print("\n[1/5] 📥 Loading historical data...")
    df = get_combined_historical_data()
    if df is None or len(df) < 100:
        print(f"❌ Not enough data: {len(df) if df else 0} records")
        print("💡 Run data collection to gather more data")
        return
    
    # 2. Train ML model (simple 24-hour prediction)
    print("\n[2/5] 🤖 Training ML model...")
    ml_model, scaler, feature_cols, ml_r2 = train_ml_model_simple(df)
    
    # 3. Generate ML forecast
    print("\n[3/5] 🔮 Generating ML forecast...")
    ml_forecasts = generate_3day_forecast_ml(ml_model, scaler, feature_cols, df)
    
    # 4. Generate Time Series forecast
    print("\n[4/5] 📈 Generating Time Series forecast...")
    ts_forecasts = generate_3day_forecast_timeseries(df)
    
    # 5. Save to MongoDB
    print("\n[5/5] 💾 Saving forecasts...")
    save_forecasts_to_mongodb(ml_forecasts, ts_forecasts)
    
    # 6. Generate report
    print("\n[6/6] 📄 Generating report...")
    generate_report(ml_forecasts, ts_forecasts)
    
    # Display summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    print("=" * 70)
    
    print("\n📊 3-DAY FORECAST SUMMARY:")
    print("-" * 50)
    print("Date         | ML Model  | Time Series | Ensemble")
    print("-" * 50)
    
    for i in range(min(len(ml_forecasts), len(ts_forecasts))):
        ml_row = ml_forecasts.iloc[i]
        ts_row = ts_forecasts.iloc[i]
        avg_aqi = (ml_row['predicted_aqi'] + ts_row['predicted_aqi']) / 2
        
        print(f"{ml_row['date']} | {ml_row['predicted_aqi']:>8.1f}  | {ts_row['predicted_aqi']:>11.1f} | {avg_aqi:>9.1f}")
    
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
    
    if ml_model is not None:
        print(f"📈 ML Model R²: {ml_r2:.4f}")
    
    print(f"📊 Reports saved in 'reports/' directory")
    print(f"💾 Forecasts saved to MongoDB collections:")
    print(f"   - ml_forecasts_3day")
    print(f"   - timeseries_forecasts_3day")
    print(f"   - ensemble_forecasts_3day")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    main()