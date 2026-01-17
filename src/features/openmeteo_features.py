"""
Feature Engineering with Open-Meteo Data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class OpenMeteoFeatureEngineer:
    """Create features from Open-Meteo data"""
    
    def __init__(self):
        self.weather_features = [
            'temperature_2m', 'relative_humidity_2m',
            'precipitation', 'pressure_msl',
            'cloud_cover', 'wind_speed_10m'
        ]
        
        self.aq_features = [
            'pm2_5', 'pm10', 'ozone', 'nitrogen_dioxide',
            'sulphur_dioxide', 'carbon_monoxide'
        ]
    
    def create_hourly_features(self, df):
        """Create hourly features from raw data"""
        features = df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in features.columns:
            if not pd.api.types.is_datetime64_any_dtype(features['timestamp']):
                features['timestamp'] = pd.to_datetime(features['timestamp'])
        
        # Sort by timestamp
        features = features.sort_values('timestamp').reset_index(drop=True)
        
        # ===== TIME FEATURES =====
        features['hour'] = features['timestamp'].dt.hour
        features['day_of_week'] = features['timestamp'].dt.dayofweek
        features['day_of_month'] = features['timestamp'].dt.day
        features['month'] = features['timestamp'].dt.month
        features['year'] = features['timestamp'].dt.year
        
        # Time of day categories
        features['time_of_day'] = pd.cut(
            features['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        
        # Day type
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        features['is_work_hours'] = ((features['hour'] >= 8) & (features['hour'] <= 18)).astype(int)
        
        # ===== WEATHER FEATURES =====
        if 'temperature_2m' in features.columns:
            features['feels_like'] = self._calculate_feels_like(
                features['temperature_2m'], 
                features.get('relative_humidity_2m', 50),
                features.get('wind_speed_10m', 0)
            )
        
        # ===== AQI FEATURES =====
        if 'pm2_5' in features.columns:
            # AQI calculation
            features['aqi_pm25'] = self._calculate_aqi_pm25(features['pm2_5'])
            
            # Lag features
            for hours in [1, 3, 6, 12, 24]:
                features[f'pm2_5_lag_{hours}h'] = features['pm2_5'].shift(hours)
            
            # Rolling statistics
            for window in [6, 12, 24]:
                features[f'pm2_5_rolling_{window}h_mean'] = features['pm2_5'].rolling(window).mean()
                features[f'pm2_5_rolling_{window}h_std'] = features['pm2_5'].rolling(window).std()
        
        if 'pm10' in features.columns:
            features['aqi_pm10'] = self._calculate_aqi_pm10(features['pm10'])
            features['pm_ratio'] = features['pm2_5'] / (features['pm10'] + 1e-6)
        
        # Combined AQI
        aqi_columns = [col for col in features.columns if col.startswith('aqi_')]
        if aqi_columns:
            features['aqi'] = features[aqi_columns].max(axis=1)
            features['aqi_category'] = features['aqi'].apply(self._get_aqi_category)
        
        # ===== INTERACTION FEATURES =====
        if 'temperature_2m' in features.columns and 'pm2_5' in features.columns:
            features['temp_pm_interaction'] = features['temperature_2m'] * features['pm2_5']
        
        if 'wind_speed_10m' in features.columns and 'pm2_5' in features.columns:
            features['wind_pm_interaction'] = features['wind_speed_10m'] * features['pm2_5']
        
        # ===== TARGET VARIABLES =====
        if 'aqi' in features.columns:
            # Next hour, next 6 hours, next 24 hours
            features['aqi_next_1h'] = features['aqi'].shift(-1)
            features['aqi_next_6h'] = features['aqi'].shift(-6)
            features['aqi_next_24h'] = features['aqi'].shift(-24)
            
            # Daily target (for next day prediction)
            features['date'] = features['timestamp'].dt.date
            daily_avg = features.groupby('date')['aqi'].transform('mean')
            features['daily_avg_aqi'] = daily_avg
            features['daily_avg_aqi_next_day'] = features.groupby('date')['aqi'].transform(
                lambda x: x.shift(-24).mean()
            )
        
        # Remove rows with NaN targets
        target_cols = [col for col in features.columns if 'next' in col or 'target' in col]
        features = features.dropna(subset=target_cols)
        
        return features
    
    def create_daily_dataset(self, hourly_features):
        """Create daily aggregated dataset for prediction"""
        if 'date' not in hourly_features.columns:
            hourly_features['date'] = hourly_features['timestamp'].dt.date
        
        # Daily aggregates
        daily_agg = hourly_features.groupby('date').agg({
            'aqi': ['mean', 'max', 'min', 'std'],
            'pm2_5': 'mean',
            'pm10': 'mean',
            'ozone': 'mean',
            'temperature_2m': ['mean', 'max', 'min'],
            'relative_humidity_2m': 'mean',
            'wind_speed_10m': 'mean',
            'precipitation': 'sum'
        }).round(2)
        
        # Flatten column names
        daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns.values]
        daily_agg = daily_agg.reset_index()
        
        # Rename columns
        daily_agg = daily_agg.rename(columns={
            'aqi_mean': 'daily_avg_aqi',
            'aqi_max': 'daily_max_aqi',
            'aqi_min': 'daily_min_aqi',
            'aqi_std': 'daily_std_aqi',
            'temperature_2m_mean': 'daily_avg_temp',
            'temperature_2m_max': 'daily_max_temp',
            'temperature_2m_min': 'daily_min_temp'
        })
        
        # Add date features
        daily_agg['date'] = pd.to_datetime(daily_agg['date'])
        daily_agg['day_of_week'] = daily_agg['date'].dt.dayofweek
        daily_agg['day_name'] = daily_agg['date'].dt.day_name()
        daily_agg['month'] = daily_agg['date'].dt.month
        daily_agg['is_weekend'] = daily_agg['day_of_week'].isin([5, 6]).astype(int)
        
        # Lag features
        daily_agg['prev_day_aqi'] = daily_agg['daily_avg_aqi'].shift(1)
        daily_agg['prev_week_aqi'] = daily_agg['daily_avg_aqi'].shift(7)
        
        # Rolling averages
        daily_agg['rolling_3d_avg_aqi'] = daily_agg['daily_avg_aqi'].rolling(3).mean()
        daily_agg['rolling_7d_avg_aqi'] = daily_agg['daily_avg_aqi'].rolling(7).mean()
        
        # Weather lag features
        daily_agg['prev_day_temp'] = daily_agg['daily_avg_temp'].shift(1)
        daily_agg['prev_day_wind'] = daily_agg['wind_speed_10m_mean'].shift(1)
        
        # Target: next day's average AQI
        daily_agg['next_day_avg_aqi'] = daily_agg['daily_avg_aqi'].shift(-1)
        
        return daily_agg
    
    def _calculate_feels_like(self, temp, humidity, wind_speed):
        """Calculate feels-like temperature (heat index approximation)"""
        # Simplified heat index calculation
        temp_c = temp
        rh = humidity
        
        # Only calculate for temperatures above 20°C
        if temp_c < 20:
            return temp_c
        
        # Heat index approximation
        c1 = -8.78469475556
        c2 = 1.61139411
        c3 = 2.33854883889
        c4 = -0.14611605
        c5 = -0.012308094
        c6 = -0.0164248277778
        c7 = 0.002211732
        c8 = 0.00072546
        c9 = -0.000003582
        
        hi = (c1 + c2 * temp_c + c3 * rh + c4 * temp_c * rh +
              c5 * temp_c**2 + c6 * rh**2 + c7 * temp_c**2 * rh +
              c8 * temp_c * rh**2 + c9 * temp_c**2 * rh**2)
        
        return hi
    
    def _calculate_aqi_pm25(self, pm25):
        """Calculate AQI for PM2.5 (US EPA)"""
        return np.where(
            pm25 <= 12.0, pm25 * (50/12.0),
            np.where(pm25 <= 35.4, 50 + (pm25 - 12.1) * (50/23.3),
            np.where(pm25 <= 55.4, 100 + (pm25 - 35.5) * (50/19.9),
            np.where(pm25 <= 150.4, 150 + (pm25 - 55.5) * (50/94.9),
            np.where(pm25 <= 250.4, 200 + (pm25 - 150.5) * (100/99.9),
                     300 + (pm25 - 250.5) * (200/249.9))))))
    
    def _calculate_aqi_pm10(self, pm10):
        """Calculate AQI for PM10"""
        return np.where(
            pm10 <= 54, pm10 * (50/54),
            np.where(pm10 <= 154, 51 + (pm10 - 55) * (49/99),
            np.where(pm10 <= 254, 101 + (pm10 - 155) * (49/99),
            np.where(pm10 <= 354, 151 + (pm10 - 255) * (49/99),
            np.where(pm10 <= 424, 201 + (pm10 - 355) * (99/69),
                     301 + (pm10 - 425) * (199/179))))))
    
    def _get_aqi_category(self, aqi):
        """Get AQI category"""
        if pd.isna(aqi):
            return "Unknown"
        elif aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

def test_feature_engineering():
    """Test the feature engineering"""
    print("🧪 Testing Open-Meteo feature engineering...")
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'pm2_5': np.random.uniform(10, 100, 100),
        'pm10': np.random.uniform(20, 150, 100),
        'temperature_2m': np.random.uniform(15, 35, 100),
        'relative_humidity_2m': np.random.uniform(30, 90, 100),
        'wind_speed_10m': np.random.uniform(0, 15, 100)
    })
    
    engineer = OpenMeteoFeatureEngineer()
    
    # Create hourly features
    hourly_features = engineer.create_hourly_features(sample_data)
    print(f"✅ Hourly features: {hourly_features.shape}")
    print(f"   Features created: {len(hourly_features.columns)}")
    
    # Create daily dataset
    daily_data = engineer.create_daily_dataset(hourly_features)
    print(f"✅ Daily dataset: {daily_data.shape}")
    print(f"   Sample columns: {list(daily_data.columns[:10])}")
    
    return hourly_features, daily_data

if __name__ == "__main__":
    test_feature_engineering()
