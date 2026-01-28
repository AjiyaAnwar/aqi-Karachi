"""
AQI Karachi - FIXED MODEL TRAINING (No ObjectId Error)
Complete solution with proper data cleaning
"""
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
import pickle
import json
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

load_dotenv()

def load_and_clean_data():
    """Load and properly clean data from MongoDB"""
    print("📥 Loading and cleaning data from MongoDB...")
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE')]
    
    # Get all data - EXCLUDE _id field which is ObjectId
    cursor = db.aqi_features.find({}, {'_id': 0})  # Exclude ObjectId
    df = pd.DataFrame(list(cursor))
    client.close()
    
    if df.empty:
        print("❌ No data found!")
        return None
    
    print(f"✅ Loaded {len(df)} records")
    
    # Convert timestamp to datetime
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
    
    # Basic data cleaning
    print("\n🧹 Data Cleaning:")
    print(f"  Removing duplicates: {df.duplicated().sum()} rows")
    df = df.drop_duplicates()
    
    # Remove outliers (AQI > 500 is impossible)
    initial_len = len(df)
    df = df[(df['aqi'] >= 0) & (df['aqi'] <= 500)]
    removed_outliers = initial_len - len(df)
    if removed_outliers > 0:
        print(f"  Removed outliers: {removed_outliers} rows")
    
    print(f"  Final data size: {len(df)} rows")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"📊 AQI stats - Min: {df['aqi'].min():.1f}, Max: {df['aqi'].max():.1f}, Mean: {df['aqi'].mean():.1f}, Std: {df['aqi'].std():.1f}")
    
    return df

def create_smart_features(df, prediction_horizon=3):
    """
    Create intelligent features for time series prediction
    Using 3-hour prediction (easier than 6 or 24)
    """
    print(f"\n🔧 Creating features for {prediction_horizon}-hour prediction...")
    
    # Make a copy
    df = df.copy()
    
    # 1. TARGET VARIABLE - Predict future AQI
    df['target'] = df['aqi'].shift(-prediction_horizon)
    
    # Remove rows without target
    df = df.dropna(subset=['target'])
    
    # 2. BASIC TIME FEATURES
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # 3. LAG FEATURES (Most important!)
    # Recent lags (1, 2, 3 hours)
    for lag in [1, 2, 3]:
        df[f'lag_{lag}h'] = df['aqi'].shift(lag)
    
    # Same time yesterday (24h ago)
    df['lag_24h'] = df['aqi'].shift(24)
    
    # 4. ROLLING STATISTICS (Simple ones)
    df['rolling_mean_3h'] = df['aqi'].rolling(window=3, min_periods=1).mean()
    df['rolling_mean_6h'] = df['aqi'].rolling(window=6, min_periods=1).mean()
    
    # 5. TIME OF DAY FEATURES
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 10)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 6. SIMPLE CYCLICAL ENCODING
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # 7. AQI TREND
    df['trend_3h'] = df['aqi'].diff(3)  # Change over 3 hours
    
    # 8. If PM data exists, add simple PM features
    if 'pm25' in df.columns:
        df['pm25_lag_1h'] = df['pm25'].shift(1)
        df['pm25_trend'] = df['pm25'].diff(1)
    
    if 'pm10' in df.columns:
        df['pm10_lag_1h'] = df['pm10'].shift(1)
    
    # 9. Remove any remaining NaN
    initial_len = len(df)
    df = df.dropna()
    removed = initial_len - len(df)
    
    print(f"✅ Removed {removed} rows with NaN, keeping {len(df)} rows")
    print(f"📋 Features created: {len(df.columns) - 3}")  # minus timestamp, target, aqi
    
    return df

def prepare_data_for_training(df):
    """Prepare clean data for training"""
    
    print("\n📊 Preparing data for training...")
    
    # Sort by time
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Select features (all numeric columns except timestamp and target)
    exclude_cols = ['timestamp', 'target', 'aqi']
    feature_cols = [col for col in df.columns 
                    if col not in exclude_cols 
                    and pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"Selected {len(feature_cols)} numeric features")
    
    # Ensure all features are numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any NaN with column mean
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
    
    # Time-based split (80/20)
    split_idx = int(len(df) * 0.8)
    
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    X_train = train[feature_cols]
    y_train = train['target']
    X_test = test[feature_cols]
    y_test = test['target']
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  Features: {len(feature_cols)}")
    
    # Time validation
    max_train_time = train['timestamp'].max()
    min_test_time = test['timestamp'].min()
    print(f"  Train period: {train['timestamp'].min().date()} to {max_train_time.date()}")
    print(f"  Test period:  {min_test_time.date()} to {test['timestamp'].max().date()}")
    
    if min_test_time <= max_train_time:
        print("⚠️ Warning: Time overlap detected!")
    else:
        print("✅ Clean time split")
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_simple_models(X_train, X_test, y_train, y_test, feature_cols):
    """Train simple models that work"""
    
    print("\n🤖 Training Simple Models...")
    
    # Convert to numpy arrays to avoid any pandas issues
    X_train_np = X_train.values.astype(np.float64)
    X_test_np = X_test.values.astype(np.float64)
    y_train_np = y_train.values.astype(np.float64)
    y_test_np = y_test.values.astype(np.float64)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled = scaler.transform(X_test_np)
    
    # Simple model configurations
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (alpha=1)': Ridge(alpha=1.0, random_state=42),
        'Ridge (alpha=10)': Ridge(alpha=10.0, random_state=42),
        'Lasso (alpha=0.1)': Lasso(alpha=0.1, random_state=42, max_iter=5000),
        'Random Forest (simple)': RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost (simple)': XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"  🏃 Training {name}...")
        
        try:
            # Train
            model.fit(X_train_scaled, y_train_np)
            
            # Predict
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(y_train_np, y_pred_train)
            test_r2 = r2_score(y_test_np, y_pred_test)
            test_mae = mean_absolute_error(y_test_np, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_test))
            
            # Store results
            metrics = {
                'Model': name,
                'Train_R2': round(train_r2, 4),
                'Test_R2': round(test_r2, 4),
                'Test_MAE': round(test_mae, 2),
                'Test_RMSE': round(test_rmse, 2)
            }
            
            results.append(metrics)
            trained_models[name] = (model, scaler, feature_cols)
            
            # Print with color coding
            if test_r2 > 0:
                print(f"    ✅ Test R²: {test_r2:.4f}, MAE: {test_mae:.2f}")
            else:
                print(f"    ⚠️ Test R²: {test_r2:.4f}, MAE: {test_mae:.2f}")
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)[:50]}")
            continue
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_R2', ascending=False)
    
    return results_df, trained_models, (X_train_scaled, X_test_scaled, y_train_np, y_test_np)

def analyze_results(results_df, trained_models, X_train_scaled, y_train):
    """Analyze and visualize results"""
    
    print("\n" + "=" * 60)
    print("📊 MODEL PERFORMANCE RESULTS")
    print("=" * 60)
    
    # Display results
    display_cols = ['Model', 'Test_R2', 'Test_MAE', 'Test_RMSE']
    display_df = results_df[display_cols].copy()
    
    # Add color highlighting
    def highlight_r2(val):
        if val > 0.7:
            return 'background-color: #d4edda'  # Green
        elif val > 0:
            return 'background-color: #fff3cd'  # Yellow
        else:
            return 'background-color: #f8d7da'  # Red
    
    styled_df = display_df.style.applymap(highlight_r2, subset=['Test_R2'])
    
    print(display_df.to_string(index=False))
    
    # Analysis
    best_r2 = results_df['Test_R2'].max()
    worst_r2 = results_df['Test_R2'].min()
    
    print(f"\n📈 Best R²: {best_r2:.4f}")
    print(f"📉 Worst R²: {worst_r2:.4f}")
    
    if best_r2 > 0:
        best_model_name = results_df.iloc[0]['Model']
        print(f"🏆 Best Model: {best_model_name}")
        
        # Feature importance for tree-based models
        if best_model_name in ['Random Forest (simple)', 'XGBoost (simple)']:
            print("\n🔍 Feature Importance Analysis:")
            model, scaler, feature_cols = trained_models[best_model_name]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                print("Top 5 features:")
                for idx, row in importance_df.head(5).iterrows():
                    print(f"  {row['Feature']}: {row['Importance']:.4f}")
                
                # Save
                os.makedirs('results', exist_ok=True)
                importance_df.to_csv('results/feature_importance.csv', index=False)
    else:
        print("\n⚠️ ALL MODELS HAVE NEGATIVE R²")
        print("Possible issues:")
        print("1. Target variable has very low variance")
        print("2. Features are not predictive")
        print("3. Data leakage issue")
    
    return best_r2

def generate_simple_forecast(trained_models, df, feature_cols, hours_ahead=24):
    """Generate simple forecast"""
    
    print(f"\n🔮 Generating {hours_ahead}-hour forecast...")
    
    # Get best model
    best_model_name = list(trained_models.keys())[0]
    model, scaler, _ = trained_models[best_model_name]
    
    # Get latest data
    latest = df.iloc[-1:].copy()
    
    forecasts = []
    current_time = datetime.now()
    
    for i in range(1, hours_ahead + 1, 3):  # Forecast every 3 hours
        forecast_time = current_time + timedelta(hours=i)
        
        # Prepare features for this forecast time
        forecast_features = {}
        
        # Time features
        forecast_features['hour'] = forecast_time.hour
        forecast_features['day_of_week'] = forecast_time.weekday()
        
        # Time of day features
        forecast_features['is_morning_rush'] = 1 if 7 <= forecast_time.hour <= 10 else 0
        forecast_features['is_evening_rush'] = 1 if 16 <= forecast_time.hour <= 19 else 0
        forecast_features['is_night'] = 1 if 0 <= forecast_time.hour <= 5 else 0
        forecast_features['is_weekend'] = 1 if forecast_time.weekday() >= 5 else 0
        
        # Cyclical encoding
        forecast_features['hour_sin'] = np.sin(2 * np.pi * forecast_time.hour / 24)
        forecast_features['hour_cos'] = np.cos(2 * np.pi * forecast_time.hour / 24)
        
        # For lag features, use average of similar hours
        similar_hours = df[df['hour'] == forecast_time.hour]
        if len(similar_hours) > 0:
            avg_aqi = similar_hours['aqi'].mean()
        else:
            avg_aqi = df['aqi'].mean()
        
        # Set lag features
        forecast_features['lag_1h'] = avg_aqi
        forecast_features['lag_2h'] = avg_aqi
        forecast_features['lag_3h'] = avg_aqi
        forecast_features['lag_24h'] = avg_aqi
        
        # Rolling stats
        forecast_features['rolling_mean_3h'] = avg_aqi
        forecast_features['rolling_mean_6h'] = avg_aqi
        
        # Other features (use averages)
        for col in feature_cols:
            if col not in forecast_features:
                if col in df.columns:
                    forecast_features[col] = df[col].mean()
                else:
                    forecast_features[col] = 0
        
        # Create feature array in correct order
        feature_array = []
        for col in feature_cols:
            feature_array.append(forecast_features.get(col, 0))
        
        feature_array = np.array(feature_array).reshape(1, -1).astype(np.float64)
        
        # Scale and predict
        features_scaled = scaler.transform(feature_array)
        predicted_aqi = float(model.predict(features_scaled)[0])
        
        # Ensure realistic range
        predicted_aqi = max(0, min(500, predicted_aqi))
        
        # AQI category
        if predicted_aqi <= 50:
            category = "Good"
        elif predicted_aqi <= 100:
            category = "Moderate"
        elif predicted_aqi <= 150:
            category = "Unhealthy for Sensitive"
        elif predicted_aqi <= 200:
            category = "Unhealthy"
        else:
            category = "Very Unhealthy"
        
        forecasts.append({
            'timestamp': forecast_time,
            'date': forecast_time.date(),
            'hour': forecast_time.hour,
            'predicted_aqi': round(predicted_aqi, 1),
            'category': category
        })
    
    forecast_df = pd.DataFrame(forecasts)
    
    print("\n📅 Forecast:")
    for _, row in forecast_df.iterrows():
        print(f"  {row['date']} {row['hour']:02d}:00 - AQI: {row['predicted_aqi']:.1f} ({row['category'][:15]})")
    
    # Save forecast
    forecast_df.to_csv('results/forecast.csv', index=False)
    print(f"💾 Forecast saved to 'results/forecast.csv'")
    
    return forecast_df

def save_artifacts(trained_models, results_df, feature_cols):
    """Save models and results"""
    
    print("\n💾 Saving artifacts...")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Save results
    results_df.to_csv('results/model_results.csv', index=False)
    print("✅ Results saved to 'results/model_results.csv'")
    
    # Save best model
    best_row = results_df.iloc[0]
    best_model_name = best_row['Model']
    
    if best_model_name in trained_models:
        model, scaler, _ = trained_models[best_model_name]
        
        model_id = f"aqi_best_{datetime.now().strftime('%Y%m%d_%H%M')}"
        model_path = f'models/{model_id}.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': scaler,
                'features': feature_cols,
                'metadata': {
                    'model_id': model_id,
                    'model_name': best_model_name,
                    'test_r2': best_row['Test_R2'],
                    'test_mae': best_row['Test_MAE'],
                    'created_at': datetime.now()
                }
            }, f)
        
        print(f"✅ Best model saved to '{model_path}'")
        
        # Save to MongoDB
        try:
            client = MongoClient(os.getenv('MONGODB_URI'))
            db = client[os.getenv('MONGODB_DATABASE')]
            
            model_record = {
                'model_id': model_id,
                'model_name': best_model_name,
                'model_type': 'ml',
                'version': '1.0',
                'created_at': datetime.now(),
                'metrics': best_row.to_dict(),
                'features_used': feature_cols,
                'is_production': best_row['Test_R2'] > 0.5
            }
            
            db.model_registry.insert_one(model_record)
            client.close()
            
            print("✅ Model saved to MongoDB")
            
        except Exception as e:
            print(f"⚠️ Could not save to MongoDB: {e}")
    
    return best_row['Test_R2']

def main():
    """Main training pipeline - SIMPLE AND ROBUST"""
    
    print("=" * 60)
    print("🌫️ AQI KARACHI - SIMPLE MODEL TRAINING")
    print("=" * 60)
    print("Starting with basic models to ensure positive R²")
    print("-" * 60)
    
    start_time = datetime.now()
    
    # 1. Load and clean data
    df_raw = load_and_clean_data()
    if df_raw is None or len(df_raw) < 50:
        print("❌ Not enough data")
        return
    
    # 2. Create simple features (3-hour prediction)
    df_features = create_smart_features(df_raw, prediction_horizon=3)
    
    if df_features is None or len(df_features) < 40:
        print("❌ Not enough data after feature creation")
        return
    
    # 3. Prepare for training
    X_train, X_test, y_train, y_test, feature_cols = prepare_data_for_training(df_features)
    
    print(f"\n📋 Features to use: {feature_cols}")
    
    # 4. Train simple models
    results_df, trained_models, scaled_data = train_simple_models(
        X_train, X_test, y_train, y_test, feature_cols
    )
    
    # 5. Analyze results
    best_r2 = analyze_results(results_df, trained_models, scaled_data[0], scaled_data[2])
    
    # 6. If we have positive R², generate forecast
    if best_r2 > 0 and len(trained_models) > 0:
        forecast_df = generate_simple_forecast(trained_models, df_features, feature_cols, hours_ahead=24)
        
        # 7. Save everything
        final_r2 = save_artifacts(trained_models, results_df, feature_cols)
    else:
        print("\n⚠️ Cannot generate forecast (negative R² or no trained models)")
        
        # Diagnostic
        print("\n🔍 DIAGNOSTIC INFORMATION:")
        print(f"Data size: {len(df_raw)}")
        print(f"Features created: {len(feature_cols)}")
        print(f"Target stats - Mean: {y_train.mean():.1f}, Std: {y_train.std():.1f}")
        
        if y_train.std() < 5:
            print("⚠️ Target has very low variance - try predicting change instead of absolute value")
            print("💡 Modify: Predict 'aqi_change_3h' instead of 'aqi'")
    
    # 8. Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"⏱️  Time: {elapsed:.1f} seconds")
    print(f"📊 Models trained: {len(trained_models)}")
    
    if best_r2 > 0:
        print(f"🏆 Best R²: {best_r2:.4f}")
        if best_r2 > 0.5:
            print("🎉 Excellent performance!")
        elif best_r2 > 0:
            print("👍 Good start!")
    else:
        print("⚠️ Need to improve model performance")
        print("\n💡 Next steps:")
        print("1. Collect more data (run data collection)")
        print("2. Try predicting AQI change instead of absolute AQI")
        print("3. Check feature engineering")

if __name__ == "__main__":
    main()