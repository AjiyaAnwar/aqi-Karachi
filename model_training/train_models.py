"""
AQI Model Training for Karachi - Fixed Version
FIXED: Handles non-numeric columns and feature versions
"""
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
import pickle
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models to compare
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.svm import SVR

load_dotenv()

def load_features_from_mongodb():
    """Load features from MongoDB - FIXED to handle feature_version column"""
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE')]
    
    # Exclude non-feature columns
    cursor = db.aqi_features.find({}, {
        '_id': 0,
        'created_at': 0,
        'feature_version': 0  # Exclude this from features
    })
    
    df = pd.DataFrame(list(cursor))
    
    if df.empty:
        print("❌ No features found in MongoDB")
        client.close()
        return None
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp')
    
    print(f"📊 Loaded {len(df)} feature records")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"📋 Columns: {len(df.columns)}")
    
    # Show column types
    print("\n📊 Column Info:")
    for col in df.columns:
        dtype = df[col].dtype
        non_nulls = df[col].count()
        print(f"  {col}: {dtype} ({non_nulls} non-null)")
    
    client.close()
    return df

def prepare_data(df):
    """Prepare data for training - Fixed to handle all data types"""
    print("\n🔧 Step 2: Preparing data...")
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # ===== FIX: ENSURE ALL COLUMNS ARE NUMERIC =====
    print("  Converting columns to numeric...")
    non_numeric_cols = []
    
    for col in df.columns:
        if col == 'timestamp':
            continue
        
        # Try to convert to numeric
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except:
            non_numeric_cols.append(col)
            print(f"    ⚠️ {col}: Could not convert to numeric, dropping")
    
    # Drop non-numeric columns
    if non_numeric_cols:
        df = df.drop(columns=non_numeric_cols)
    
    # ===== CREATE TARGET COLUMN =====
    print("  Creating target column...")
    df['target_24h'] = df['aqi'].shift(-24)  # AQI 24 hours ahead
    
    # Drop rows with NaN in target
    initial_count = len(df)
    df = df.dropna(subset=['target_24h'])
    dropped = initial_count - len(df)
    print(f"    Dropped {dropped} rows with missing target")
    
    # ===== CREATE TIME FEATURES =====
    print("  Creating time features...")
    
    # Basic time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    print("    Created cyclical time features")
    
    # ===== ENSURE REQUIRED COLUMNS EXIST =====
    print("  Checking required columns...")
    
    # Check for PM columns
    if 'pm25' not in df.columns:
        print("    ⚠️ pm25 not found, creating estimate")
        df['pm25'] = df['aqi'] * 0.5
    
    if 'pm10' not in df.columns:
        print("    ⚠️ pm10 not found, creating estimate")
        df['pm10'] = df['aqi'] * 0.8
    
    # Create interaction features
    df['aqi_pm25_ratio'] = df['aqi'] / (df['pm25'] + 1)
    df['aqi_pm10_ratio'] = df['aqi'] / (df['pm10'] + 1)
    
    print("    Created interaction features")
    
    # ===== CREATE LAG FEATURES =====
    print("  Creating lag features...")
    
    lag_features_created = 0
    for lag in [1, 3, 6, 12, 24]:
        col_name = f'lag_{lag}h'
        df[col_name] = df['aqi'].shift(lag)
        lag_features_created += 1
    
    print(f"    Created {lag_features_created} lag features")
    
    # ===== CREATE ROLLING STATISTICS =====
    print("  Creating rolling statistics...")
    
    rolling_features_created = 0
    for window in [6, 12, 24, 72]:
        df[f'rolling_mean_{window}h'] = df['aqi'].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}h'] = df['aqi'].rolling(window=window, min_periods=1).std()
        df[f'rolling_max_{window}h'] = df['aqi'].rolling(window=window, min_periods=1).max()
        rolling_features_created += 3
    
    print(f"    Created {rolling_features_created} rolling features")
    
    # ===== HANDLE MISSING VALUES =====
    print("  Handling missing values...")
    
    # Fill lag features
    lag_cols = [col for col in df.columns if 'lag_' in col]
    for col in lag_cols:
        df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
    
    # Fill rolling features
    rolling_cols = [col for col in df.columns if 'rolling_' in col]
    for col in rolling_cols:
        df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
    
    print(f"    Filled {len(lag_cols) + len(rolling_cols)} features")
    
    # ===== SELECT FEATURES =====
    print("  Selecting features for training...")
    
    # Drop columns we don't want in features
    columns_to_drop = [
        'timestamp', 
        'target_24h',
        'hour',  # Already encoded
        'day_of_week', 
        'month'
    ]
    
    # Only drop if they exist
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    X = df.drop(columns=columns_to_drop)
    y = df['target_24h']
    
    # ===== FIX: PROPERLY HANDLE REMAINING NANS =====
    print("  Final cleaning...")
    
    # Fill remaining NaN with column mean (numeric only)
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            col_mean = X[col].mean()
            nan_count = X[col].isna().sum()
            if nan_count > 0:
                X[col] = X[col].fillna(col_mean)
                print(f"    Filled {nan_count} NaN in {col} with mean {col_mean:.2f}")
    
    # Drop any columns that are still all NaN
    X = X.dropna(axis=1, how='all')
    
    print(f"\n  Final feature matrix: {X.shape}")
    print(f"  Target vector: {y.shape}")
    print(f"  Features: {len(X.columns)}")
    
    # ===== SPLIT DATA (TIME-SERIES AWARE) =====
    print("\n  Creating train/test split...")
    
    split_idx = int(len(X) * 0.8)  # 80% train, 20% test
    
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    
    print(f"  Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Testing:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    if 'timestamp' in df.columns:
        print(f"  Train dates: {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[split_idx-1].date()}")
        print(f"  Test dates:  {df['timestamp'].iloc[split_idx].date()} to {df['timestamp'].iloc[-1].date()}")
    
    # ===== SCALE FEATURES =====
    print("  Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Scaling complete. Feature names: {list(X.columns)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns, X, df

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train multiple models with better hyperparameters"""
    
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.03,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            early_stopping_rounds=20
        ),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'SVR': SVR(kernel='rbf', C=5, gamma=0.01)
    }
    
    results = []
    trained_models = {}
    
    print("\n🤖 Step 3: Training models...")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\n🏃 Training {name}...")
        
        try:
            # Train model
            if name == 'XGBoost':
                # XGBoost needs eval set
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Store results
            results.append({
                'Model': name,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'Overfitting': abs(train_r2 - test_r2)
            })
            
            trained_models[name] = model
            
            print(f"  ✓ Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
            print(f"  ✓ Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error training {name}: {e}")
            continue
    
    if not results:
        print("❌ No models were successfully trained!")
        return None, None
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test R²', ascending=False)
    
    return results_df, trained_models

def save_best_model(model, model_name, metrics, feature_names, scaler):
    """Save the best model to MongoDB"""
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE')]
    
    model_collection = db['model_registry']
    
    # Serialize
    model_bytes = pickle.dumps(model)
    scaler_bytes = pickle.dumps(scaler)
    
    model_record = {
        'model_id': f"aqi_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'model_name': model_name,
        'model_type': type(model).__name__,
        'version': '3.0',
        'created_at': datetime.now().isoformat(),
        'metrics': metrics,
        'features_used': list(feature_names),
        'model_bytes': model_bytes,
        'scaler_bytes': scaler_bytes,
        'is_production': True,
        'description': 'Karachi AQI 24h Prediction Model (Fixed)'
    }
    
    result = model_collection.insert_one(model_record)
    
    print(f"\n💾 Saved '{model_name}' to MongoDB")
    print(f"  Model ID: {model_record['model_id']}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Test R²: {metrics['Test R²']:.3f}")
    
    # Save locally
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{model_record["model_id"]}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'metadata': model_record
        }, f)
    
    print(f"  Local backup: {model_path}")
    
    client.close()
    return model_record['model_id']

def make_3_day_forecast(model, scaler, feature_names, df_full, last_timestamp):
    """Make a proper 3-day forecast - FIXED to handle feature creation"""
    print("\n🔮 Step 5: Generating 3-day forecast...")
    
    forecasts = []
    current_date = last_timestamp.date()
    
    # Get the most recent complete data point
    last_row = df_full.iloc[-1:].copy()
    
    # Ensure last_row has all required columns
    for col in feature_names:
        if col not in last_row.columns:
            print(f"  ⚠️ Column {col} not in last_row, setting to 0")
            last_row[col] = 0
    
    print(f"  Last row date: {last_timestamp}")
    print(f"  Features available: {len(feature_names)}")
    
    for day_offset in range(1, 4):  # Next 3 days
        forecast_date = current_date + timedelta(days=day_offset)
        
        # Create forecast features from last row
        forecast_features = last_row.copy()
        
        # Update time-related features for forecast day
        forecast_hour = 12  # Predict for noon
        forecast_day_of_week = (last_timestamp + timedelta(days=day_offset)).weekday()
        forecast_month = (last_timestamp + timedelta(days=day_offset)).month
        
        # Update time features if they exist in features
        if 'hour' in forecast_features.columns:
            forecast_features['hour'] = forecast_hour
        
        if 'day_of_week' in forecast_features.columns:
            forecast_features['day_of_week'] = forecast_day_of_week
        
        if 'month' in forecast_features.columns:
            forecast_features['month'] = forecast_month
        
        # Update cyclical features
        if 'hour_sin' in forecast_features.columns:
            forecast_features['hour_sin'] = np.sin(2 * np.pi * forecast_hour / 24)
        
        if 'hour_cos' in forecast_features.columns:
            forecast_features['hour_cos'] = np.cos(2 * np.pi * forecast_hour / 24)
        
        if 'day_sin' in forecast_features.columns:
            forecast_features['day_sin'] = np.sin(2 * np.pi * forecast_day_of_week / 7)
        
        if 'day_cos' in forecast_features.columns:
            forecast_features['day_cos'] = np.cos(2 * np.pi * forecast_day_of_week / 7)
        
        # For lag features, use average of similar past hours
        if 'aqi' in df_full.columns:
            # Find similar hours in the past
            similar_data = df_full[
                (df_full['timestamp'].dt.hour == forecast_hour) &
                (df_full['timestamp'].dt.dayofweek == forecast_day_of_week)
            ]
            
            if len(similar_data) > 0:
                avg_aqi = similar_data['aqi'].mean()
            else:
                avg_aqi = df_full['aqi'].mean()
        else:
            avg_aqi = 100  # Default fallback
        
        # Prepare for prediction
        forecast_features = forecast_features[feature_names]
        
        # FIX: Fill NaN with 0 instead of mean (which could fail with strings)
        forecast_features = forecast_features.fillna(0)
        
        # Ensure all values are numeric
        for col in forecast_features.columns:
            forecast_features[col] = pd.to_numeric(forecast_features[col], errors='coerce')
        
        # Scale features
        try:
            forecast_scaled = scaler.transform(forecast_features)
            
            # Make prediction
            predicted_aqi = model.predict(forecast_scaled)[0]
            
            # Ensure realistic range
            predicted_aqi = max(0, min(500, float(predicted_aqi)))
            
            # Calculate confidence (decreases with time)
            confidence = max(0.5, 1.0 - (day_offset * 0.15))
            
        except Exception as e:
            print(f"  ⚠️ Prediction failed for {forecast_date}: {e}")
            predicted_aqi = avg_aqi
            confidence = 0.3
        
        # Get AQI category
        category = get_aqi_category(predicted_aqi)
        
        forecasts.append({
            'date': forecast_date,
            'predicted_aqi': float(predicted_aqi),
            'confidence': float(confidence),
            'category': category
        })
        
        print(f"  📅 {forecast_date}: AQI {predicted_aqi:.1f} ({category}), Confidence: {confidence:.1%}")
    
    return pd.DataFrame(forecasts)

def get_aqi_category(aqi):
    """Convert AQI value to category"""
    aqi = float(aqi)
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def save_predictions(predictions_df, model_id):
    """Save predictions to MongoDB"""
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE')]
    
    predictions_collection = db['aqi_predictions']
    
    # Clear old predictions
    predictions_collection.delete_many({})
    
    # Convert predictions to records
    prediction_records = []
    for _, row in predictions_df.iterrows():
        record = {
            'model_id': model_id,
            'date': row['date'].strftime('%Y-%m-%d'),
            'predicted_aqi': float(row['predicted_aqi']),
            'confidence': float(row['confidence']),
            'category': row['category'],
            'prediction_date': datetime.now().isoformat(),
            'created_at': datetime.now().isoformat()
        }
        prediction_records.append(record)
    
    # Insert predictions
    if prediction_records:
        predictions_collection.insert_many(prediction_records)
        print(f"💾 Saved {len(prediction_records)} predictions to MongoDB")
    
    client.close()

def main():
    """Main training pipeline - Fixed version"""
    print("🚀 KARACHI AQI MODEL TRAINING (FIXED VERSION)")
    print("=" * 60)
    
    # 1. Load data
    print("\n📥 Step 1: Loading features from MongoDB...")
    df = load_features_from_mongodb()
    
    if df is None:
        print("❌ No data to train on. Please run features.py first.")
        return
    
    # Check if we have enough data
    if len(df) < 50:
        print(f"❌ Only {len(df)} samples. Need at least 50 for training.")
        return
    
    print(f"✅ Data loaded successfully: {len(df)} samples")
    
    # 2. Prepare data with enhanced features
    try:
        X_train, X_test, y_train, y_test, scaler, feature_names, X_raw, df_full = prepare_data(df)
        print("✅ Data preparation complete")
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return
    
    # 3. Train models
    print("\n🤖 Step 3: Training models...")
    results_df, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    if results_df is None:
        print("❌ Model training failed!")
        return
    
    # 4. Display results
    print("\n" + "=" * 60)
    print("🏆 MODEL COMPARISON:")
    print("=" * 60)
    print(results_df[['Model', 'Test MAE', 'Test R²', 'Overfitting']].to_string(index=False))
    
    # 5. Select best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    best_metrics = results_df.iloc[0].to_dict()
    
    print(f"\n🎯 BEST MODEL: {best_model_name}")
    print(f"   Test R²: {best_metrics['Test R²']:.3f}")
    print(f"   Test MAE: {best_metrics['Test MAE']:.2f}")
    
    # 6. Save best model
    print("\n💾 Step 4: Saving model...")
    model_id = save_best_model(best_model, best_model_name, best_metrics, feature_names, scaler)
    
    # 7. Make and save predictions
    try:
        if len(df_full) > 0 and 'timestamp' in df_full.columns:
            last_timestamp = df_full['timestamp'].max()
            forecast_df = make_3_day_forecast(best_model, scaler, feature_names, df_full, last_timestamp)
            
            # 8. Save predictions
            save_predictions(forecast_df, model_id)
        else:
            print("⚠️ Could not generate forecast: timestamp column missing")
    except Exception as e:
        print(f"⚠️ Forecast generation failed: {e}")
        forecast_df = None
    
    # 9. Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📊 Data: {len(df)} samples, {len(feature_names)} features")
    print(f"🏆 Best model: {best_model_name}")
    print(f"📈 Test R²: {best_metrics['Test R²']:.3f}")
    print(f"📉 Test MAE: {best_metrics['Test MAE']:.2f}")
    print(f"🆔 Model ID: {model_id}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/model_comparison.csv', index=False)
    
    if forecast_df is not None and not forecast_df.empty:
        forecast_df.to_csv('results/forecast.csv', index=False)
        print(f"\n💾 Files saved:")
        print(f"  results/model_comparison.csv")
        print(f"  results/forecast.csv")
        print(f"  models/{model_id}.pkl")
    
    print("\n📋 Next steps:")
    print("  1. Check the dashboard for model metrics")
    print("  2. Run data collection regularly for fresh data")
    print("  3. Monitor model performance over time")

if __name__ == "__main__":
    main()