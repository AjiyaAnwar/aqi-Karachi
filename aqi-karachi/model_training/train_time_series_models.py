"""
FIXED TIME SERIES FORECASTING - Generate DAILY forecasts for 3 days
Uses: seasonal + exponential smoothing + moving average
Saves DAILY forecasts to: timeseries_forecasts_3day
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

def get_historical_data():
    """Get historical AQI data"""
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE', 'aqi_predictor')]
    
    cursor = db.aqi_measurements.find({}, {'_id': 0, 'timestamp': 1, 'aqi': 1})
    df = pd.DataFrame(list(cursor))
    client.close()
    
    if df.empty:
        return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    return df

def simple_seasonal_forecast(df, days=3):
    """Simple seasonal forecast for NEXT 3 DAYS"""
    print("🤖 Creating seasonal forecast for next 3 days...")
    
    # Extract hour of day
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Calculate average by hour and day type
    hourly_avg = df.groupby('hour')['aqi'].mean()
    weekday_avg = df[df['day_of_week'] < 5]['aqi'].mean()
    weekend_avg = df[df['day_of_week'] >= 5]['aqi'].mean()
    
    # Create DAILY forecasts
    predictions = []
    today = datetime.now().date()
    
    for day_offset in range(1, days + 1):
        forecast_date = today + timedelta(days=day_offset)
        day_of_week = forecast_date.weekday()
        
        # Calculate daily average AQI
        day_predictions = []
        for hour in range(24):  # 24 hours in a day
            forecast_time = datetime.combine(forecast_date, datetime.min.time()) + timedelta(hours=hour)
            
            # Base prediction from hourly average
            base_pred = hourly_avg.get(hour, df['aqi'].mean())
            
            # Adjust for weekday/weekend
            if day_of_week >= 5:  # Weekend
                base_pred = base_pred * 0.95  # Slightly lower on weekends
            
            # Adjust for time of day
            if 8 <= hour <= 20:  # Daytime hours
                base_pred = base_pred * 1.05
            
            # Add small variation
            variation = np.random.normal(0, 2)
            predicted = max(10, base_pred + variation)
            
            day_predictions.append(predicted)
        
        # Daily average AQI
        daily_avg_aqi = np.mean(day_predictions)
        
        predictions.append({
            'timestamp': datetime.combine(forecast_date, datetime.min.time()),
            'date': forecast_date.strftime('%Y-%m-%d'),
            'predicted_aqi': daily_avg_aqi,
            'model': 'seasonal',
            'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'forecast_day': day_offset
        })
    
    return pd.DataFrame(predictions)

def exponential_smoothing_forecast(df, days=3, alpha=0.3):
    """Exponential smoothing forecast for NEXT 3 DAYS"""
    print("🤖 Creating exponential smoothing forecast for next 3 days...")
    
    # Get recent daily averages
    df['date_only'] = df['timestamp'].dt.date
    daily_avg = df.groupby('date_only')['aqi'].mean().reset_index()
    
    if len(daily_avg) < 2:
        # Not enough data, use simple average
        base_aqi = df['aqi'].mean()
    else:
        # Simple exponential smoothing
        aqi_series = daily_avg['aqi'].values
        last_value = aqi_series[-1]
        prev_value = aqi_series[-2] if len(aqi_series) > 1 else last_value
        
        # Calculate trend
        trend = last_value - prev_value
    
    # Create DAILY forecasts
    predictions = []
    today = datetime.now().date()
    
    for day_offset in range(1, days + 1):
        forecast_date = today + timedelta(days=day_offset)
        day_of_week = forecast_date.weekday()
        
        # Base prediction with trend
        if len(daily_avg) >= 2:
            predicted = last_value + (trend * alpha * day_offset)
        else:
            predicted = base_aqi
        
        # Adjust for weekday/weekend
        if day_of_week >= 5:  # Weekend
            predicted = predicted * 0.97
        
        # Add small variation
        variation = np.random.normal(0, 3)
        predicted = max(10, predicted + variation)
        
        predictions.append({
            'timestamp': datetime.combine(forecast_date, datetime.min.time()),
            'date': forecast_date.strftime('%Y-%m-%d'),
            'predicted_aqi': predicted,
            'model': 'exponential_smoothing',
            'day_of_week': day_of_week,
            'forecast_day': day_offset
        })
    
    return pd.DataFrame(predictions)

def moving_average_forecast(df, days=3):
    """Moving average forecast for NEXT 3 DAYS"""
    print("🤖 Creating moving average forecast for next 3 days...")
    
    # Get recent daily averages
    df['date_only'] = df['timestamp'].dt.date
    daily_avg = df.groupby('date_only')['aqi'].mean().reset_index()
    
    # Calculate moving averages
    if len(daily_avg) >= 7:
        ma_3d = daily_avg['aqi'].tail(3).mean()
        ma_7d = daily_avg['aqi'].tail(7).mean()
        base_pred = ma_3d * 0.6 + ma_7d * 0.4
    elif len(daily_avg) >= 3:
        ma_3d = daily_avg['aqi'].tail(3).mean()
        base_pred = ma_3d
    else:
        base_pred = df['aqi'].mean()
    
    # Create DAILY forecasts
    predictions = []
    today = datetime.now().date()
    
    for day_offset in range(1, days + 1):
        forecast_date = today + timedelta(days=day_offset)
        day_of_week = forecast_date.weekday()
        
        # Base prediction
        predicted = base_pred
        
        # Adjust for weekday/weekend
        if day_of_week >= 5:  # Weekend
            predicted = predicted * 0.96
        
        # Add weekly pattern
        if day_of_week == 0:  # Monday
            predicted = predicted * 1.02
        
        # Add small variation
        variation = np.random.normal(0, 2)
        predicted = max(10, predicted + variation)
        
        predictions.append({
            'timestamp': datetime.combine(forecast_date, datetime.min.time()),
            'date': forecast_date.strftime('%Y-%m-%d'),
            'predicted_aqi': predicted,
            'model': 'moving_average',
            'day_of_week': day_of_week,
            'forecast_day': day_offset
        })
    
    return pd.DataFrame(predictions)

def save_time_series_forecasts(predictions_df):
    """
    Save time series forecasts to CORRECT collections
    ONLY saves to: timeseries_forecasts_3day
    """
    if predictions_df.empty:
        return
    
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DATABASE', 'aqi_predictor')]
    
    # DELETE old forecasts from SINGLE collection
    db.timeseries_forecasts_3day.delete_many({})
    print("🗑️  Cleared old time series forecasts")
    
    # Prepare records
    records = []
    for _, row in predictions_df.iterrows():
        timestamp = row['timestamp']
        date_str = row['date']
        
        record = {
            'date': date_str,
            'predicted_aqi': float(round(row['predicted_aqi'], 1)),
            'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
            'model': 'timeseries',
            'model_type': row.get('model', 'ensemble'),
            'created_at': datetime.now(),
            'day_of_week': row.get('day_of_week', 0),
            'is_weekend': row.get('is_weekend', 0),
            'forecast_day': row.get('forecast_day', 0)
        }
        records.append(record)
    
    # Save to SINGLE collection
    if records:
        db.timeseries_forecasts_3day.insert_many(records)
        print(f"✅ Saved {len(records)} DAILY time series forecasts to 'timeseries_forecasts_3day'")
    
    client.close()

def create_ensemble_forecast(all_predictions):
    """Create ensemble from all time series methods"""
    print("🤖 Creating time series ensemble...")
    
    if all_predictions.empty:
        return pd.DataFrame()
    
    # Group by date and take weighted average
    ensemble_df = all_predictions.groupby('date').agg({
        'predicted_aqi': 'mean',
        'day_of_week': 'first'
    }).reset_index()
    
    # Add model info
    ensemble_df['model'] = 'timeseries_ensemble'
    ensemble_df['timestamp'] = pd.to_datetime(ensemble_df['date'])
    
    return ensemble_df

def main():
    """Main time series forecasting for NEXT 3 DAYS"""
    print("=" * 60)
    print("📈 FIXED TIME-SERIES FORECASTING (3 DAYS)")
    print("Generating DAILY forecasts for next 3 days")
    print("=" * 60)
    
    # Load data
    df = get_historical_data()
    if df is None:
        print("❌ No data available")
        return
    
    print(f"📊 Loaded {len(df)} historical records")
    print(f"📅 From {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"📈 Current AQI: {df['aqi'].iloc[-1]:.1f}")
    
    # Generate DAILY forecasts for next 3 days
    print("\n" + "=" * 60)
    
    # 1. Seasonal forecast (3 days)
    seasonal_pred = simple_seasonal_forecast(df, days=3)
    print(f"✅ Seasonal: {len(seasonal_pred)} daily forecasts")
    
    # 2. Exponential smoothing (3 days)
    es_pred = exponential_smoothing_forecast(df, days=3)
    print(f"✅ Exponential Smoothing: {len(es_pred)} daily forecasts")
    
    # 3. Moving average (3 days)
    ma_pred = moving_average_forecast(df, days=3)
    print(f"✅ Moving Average: {len(ma_pred)} daily forecasts")
    
    # Combine all predictions
    all_predictions = pd.concat([seasonal_pred, es_pred, ma_pred], ignore_index=True)
    
    # Create ensemble (weighted average)
    ensemble_df = create_ensemble_forecast(all_predictions)
    
    # Save ALL predictions (individual methods + ensemble)
    print("\n💾 SAVING FORECASTS")
    print("-" * 40)
    
    # Save individual method predictions
    save_time_series_forecasts(all_predictions)
    
    # Show predictions
    print("\n" + "=" * 60)
    print("📅 3-DAY FORECAST SUMMARY")
    print("=" * 60)
    
    # Show ensemble predictions
    if not ensemble_df.empty:
        ensemble_df = ensemble_df.sort_values('date')
        
        day_names = ["Tomorrow", "Day 2", "Day 3"]
        
        for i, (_, row) in enumerate(ensemble_df.iterrows(), 1):
            date_str = row['date']
            aqi = row['predicted_aqi']
            day_of_week = row['day_of_week']
            
            # Format date
            forecast_date = datetime.strptime(date_str, '%Y-%m-%d')
            display_date = forecast_date.strftime('%A, %b %d')
            
            # AQI category
            if aqi <= 50:
                category = "Good"
                emoji = "😊"
                color = "🟢"
            elif aqi <= 100:
                category = "Moderate"
                emoji = "😐"
                color = "🟡"
            elif aqi <= 150:
                category = "Unhealthy for Sensitive Groups"
                emoji = "😷"
                color = "🟠"
            elif aqi <= 200:
                category = "Unhealthy"
                emoji = "🤒"
                color = "🔴"
            elif aqi <= 300:
                category = "Very Unhealthy"
                emoji = "🏥"
                color = "🟣"
            else:
                category = "Hazardous"
                emoji = "☣️"
                color = "⚫"
            
            print(f"  {color} {day_names[i-1]} ({display_date}):")
            print(f"     AQI: {aqi:.1f} {emoji}")
            print(f"     Category: {category}")
            print()
    
    print(f"\n✅ Time series forecasting complete!")
    print(f"📊 Generated {len(all_predictions)} forecast points")
    print(f"📅 {len(ensemble_df)} daily ensemble forecasts")
    print(f"💾 Saved to 'timeseries_forecasts_3day' collection")

if __name__ == "__main__":
    main()