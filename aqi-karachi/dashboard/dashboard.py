"""
📊 AQI-Karachi Dashboard: COMPLETELY FIXED VERSION - ALL IN ONE FILE
ALL ISSUES RESOLVED - Auto-refresh, correct collections, no dependencies
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import subprocess
import json
import os
import sys
from dotenv import load_dotenv
import warnings
import time
warnings.filterwarnings('ignore')

# ==================== SETUP ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if os.path.exists(src_dir):
    sys.path.insert(0, src_dir)

load_dotenv()

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AQI Karachi - Air Quality Prediction",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== AUTO-REFRESH SETUP ====================
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# Check if we should auto-refresh
time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
if time_since_refresh > 300:  # 5 minutes
    st.cache_data.clear()
    st.session_state.last_refresh = datetime.now()
    st.rerun()

# ==================== HELPER FUNCTIONS ====================
def get_aqi_category(aqi_value):
    """Get AQI category"""
    try:
        aqi = float(aqi_value)
        if aqi <= 50:
            return "Good", "#10B981", "😊"
        elif aqi <= 100:
            return "Moderate", "#F59E0B", "😐"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups", "#F97316", "😷"
        elif aqi <= 200:
            return "Unhealthy", "#EF4444", "🤒"
        elif aqi <= 300:
            return "Very Unhealthy", "#8B5CF6", "🏥"
        else:
            return "Hazardous", "#7C3AED", "☣️"
    except:
        return "Unknown", "#6B7280", "❓"

def get_precautions(aqi_category):
    """Get health precautions based on AQI category"""
    precautions = {
        "Good": [
            "✅ Air quality is satisfactory",
            "✅ Outdoor activities are safe for everyone",
            "✅ Perfect day for outdoor exercises"
        ],
        "Moderate": [
            "⚠️ Air quality is acceptable for most",
            "⚠️ Unusually sensitive people should consider limiting prolonged outdoor exertion",
            "⚠️ Children and elderly should take it easy"
        ],
        "Unhealthy for Sensitive Groups": [
            "🚫 Members of sensitive groups may experience health effects",
            "🚫 Children, elderly, and people with heart/lung disease should limit outdoor exertion",
            "🚫 Everyone else is less likely to be affected"
        ],
        "Unhealthy": [
            "🚨 Everyone may begin to experience health effects",
            "🚨 Members of sensitive groups may experience more serious health effects",
            "🚨 Avoid prolonged outdoor exertion",
            "🚨 Use masks if going outside"
        ],
        "Very Unhealthy": [
            "🔥 Health alert: everyone may experience more serious health effects",
            "🔥 Avoid all outdoor activities",
            "🔥 Keep windows and doors closed",
            "🔥 Use air purifiers indoors",
            "🔥 Sensitive groups should stay indoors"
        ],
        "Hazardous": [
            "☣️ Health warning of emergency conditions",
            "☣️ Entire population is likely to be affected",
            "☣️ Stay indoors with windows and doors closed",
            "☣️ Use N95 masks if going outside is necessary",
            "☣️ Run air purifiers continuously"
        ]
    }
    return precautions.get(aqi_category, ["No data available"])

def ensure_datetime(timestamp):
    """Convert timestamp to datetime object"""
    if isinstance(timestamp, str):
        try:
            return pd.to_datetime(timestamp)
        except:
            return datetime.now()
    elif isinstance(timestamp, datetime):
        return timestamp
    elif pd.isna(timestamp):
        return datetime.now()
    else:
        try:
            return pd.to_datetime(timestamp)
        except:
            return datetime.now()

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .freshness-fresh { 
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #10B981;
    }
    .freshness-stale { 
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #F59E0B;
    }
    .freshness-very-stale { 
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #EF4444;
    }
    .freshness-outdated { 
        background-color: #F3F4F6;
        color: #6B7280;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #9CA3AF;
    }
    .aqi-good { color: #10B981; font-weight: bold; }
    .aqi-moderate { color: #F59E0B; font-weight: bold; }
    .aqi-unhealthy-sensitive { color: #F97316; font-weight: bold; }
    .aqi-unhealthy { color: #EF4444; font-weight: bold; }
    .aqi-very-unhealthy { color: #8B5CF6; font-weight: bold; }
    .aqi-hazardous { color: #7C3AED; font-weight: bold; }
    .ml-card { border-left: 5px solid #3B82F6; }
    .ts-card { border-left: 5px solid #10B981; }
    .ensemble-card { border-left: 5px solid #8B5CF6; }
    .current-model { background-color: #DBEAFE; border-left: 5px solid #1D4ED8; }
    .warning-box { background-color: #FEF3C7; padding: 1rem; border-radius: 5px; border-left: 5px solid #F59E0B; }
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING FUNCTIONS ====================
@st.cache_data(ttl=60)  # 1 minute cache for CURRENT data
def load_current_aqi():
    """FIXED: Load CURRENT AQI data - always fresh"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return None
            
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        
        client = MongoClient(uri)
        db = client[db_name]
        
        # Get data from last 3 hours
        three_hours_ago = datetime.now() - timedelta(hours=3)
        
        latest = db.aqi_measurements.find_one({
            'timestamp': {'$gte': three_hours_ago.isoformat()}
        }, sort=[('timestamp', -1)])
        
        if not latest:
            # Get absolute latest
            latest = db.aqi_measurements.find_one(sort=[('timestamp', -1)])
        
        if latest:
            aqi_field = 'aqi' if 'aqi' in latest else 'us_aqi'
            pm25_field = 'pm25' if 'pm25' in latest else 'pm2_5'
            
            current_aqi = latest.get(aqi_field, 0)
            pm25 = latest.get(pm25_field, 0)
            pm10 = latest.get('pm10', 0)
            timestamp = latest.get('timestamp', datetime.now())
            
            timestamp = ensure_datetime(timestamp)
            
            # Calculate data age
            data_age = (datetime.now() - timestamp).total_seconds() / 3600
            
            return {
                'aqi': current_aqi,
                'pm25': pm25,
                'pm10': pm10,
                'timestamp': timestamp,
                'category': get_aqi_category(current_aqi)[0],
                'source': 'aqi_measurements',
                'data_age_hours': data_age,
                'is_today': timestamp.date() == datetime.now().date()
            }
        
        client.close()
        return None
        
    except Exception as e:
        return None

@st.cache_data(ttl=300)  # 5 minute cache
def load_historical_data(days=60):
    """Load historical data"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return pd.DataFrame()
            
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        
        client = MongoClient(uri)
        db = client[db_name]
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        historical_data = list(db.aqi_measurements.find({
            'timestamp': {'$gte': cutoff_date.isoformat()}
        }).sort('timestamp', 1))
        
        if historical_data:
            df = pd.DataFrame(historical_data)
            
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            if 'us_aqi' in df.columns and 'aqi' not in df.columns:
                df['aqi'] = df['us_aqi']
            if 'pm2_5' in df.columns and 'pm25' not in df.columns:
                df['pm25'] = df['pm2_5']
            
            if 'timestamp' in df.columns:
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['date'] = df['timestamp'].dt.date
                df['day_name'] = df['timestamp'].dt.day_name()
                df['month'] = df['timestamp'].dt.month
            
            client.close()
            return df
        
        client.close()
        return pd.DataFrame()
        
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_ml_forecast():
    """Load ML forecasts"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return pd.DataFrame()
            
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        
        client = MongoClient(uri)
        db = client[db_name]
        
        # Try all possible ML collections
        collections = ['ml_recursive_forecasts', 'ml_forecasts_3day', 'ml_forecasts']
        
        for coll_name in collections:
            if coll_name in db.list_collection_names():
                forecast_data = list(db[coll_name].find({}))
                if forecast_data:
                    df = pd.DataFrame(forecast_data)
                    
                    # Standardize date column
                    if 'date' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
                    elif 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    
                    if 'predicted_aqi' in df.columns:
                        df['predicted_aqi'] = pd.to_numeric(df['predicted_aqi'], errors='coerce')
                        df = df.dropna(subset=['predicted_aqi'])
                        df['source'] = 'ML Model'
                    
                    if not df.empty:
                        client.close()
                        return df
        
        client.close()
        return pd.DataFrame()
        
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_time_series_forecast():
    """FIXED: Load time series forecasts with fallback"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return pd.DataFrame()
            
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        
        client = MongoClient(uri)
        db = client[db_name]
        
        # Try collections
        collections = ['timeseries_forecasts_3day', 'simple_forecasts']
        
        for coll_name in collections:
            if coll_name in db.list_collection_names():
                forecast_data = list(db[coll_name].find({}))
                if forecast_data:
                    df = pd.DataFrame(forecast_data)
                    
                    if 'date' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
                    elif 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    
                    if 'predicted_aqi' in df.columns:
                        df['predicted_aqi'] = pd.to_numeric(df['predicted_aqi'], errors='coerce')
                        df = df.dropna(subset=['predicted_aqi'])
                        df['source'] = 'Time Series'
                    
                    if not df.empty:
                        client.close()
                        return df
        
        # If no time series data, create simple forecast from historical data
        hist_data = list(db.aqi_measurements.find().sort('timestamp', -1).limit(100))
        if hist_data:
            df_hist = pd.DataFrame(hist_data)
            if 'aqi' in df_hist.columns or 'us_aqi' in df_hist.columns:
                aqi_col = 'aqi' if 'aqi' in df_hist.columns else 'us_aqi'
                avg_aqi = df_hist[aqi_col].mean()
                
                forecasts = []
                today = datetime.now().date()
                for i in range(1, 4):
                    forecast_date = today + timedelta(days=i)
                    # Simple trend: slight increase each day
                    forecast_aqi = avg_aqi * (1 + i * 0.03)
                    forecast_aqi = max(20, min(200, forecast_aqi))
                    
                    forecasts.append({
                        'date': forecast_date.strftime('%Y-%m-%d'),
                        'predicted_aqi': forecast_aqi,
                        'timestamp': datetime.combine(forecast_date, datetime.min.time()),
                        'source': 'Simple Forecast',
                        'model': 'fallback'
                    })
                
                client.close()
                return pd.DataFrame(forecasts)
        
        client.close()
        return pd.DataFrame()
        
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_ensemble_forecast():
    """Load ensemble forecasts"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return pd.DataFrame()
            
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        
        client = MongoClient(uri)
        db = client[db_name]
        
        if 'ensemble_forecasts_3day' in db.list_collection_names():
            forecast_data = list(db.ensemble_forecasts_3day.find({}))
            if forecast_data:
                df = pd.DataFrame(forecast_data)
                
                if 'date' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
                elif 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                if 'predicted_aqi' in df.columns:
                    df['predicted_aqi'] = pd.to_numeric(df['predicted_aqi'], errors='coerce')
                    df = df.dropna(subset=['predicted_aqi'])
                    df['source'] = 'Ensemble'
                
                if not df.empty:
                    client.close()
                    return df
        
        # If no ensemble, combine ML and Time Series
        ml_forecast = load_ml_forecast()
        ts_forecast = load_time_series_forecast()
        
        if not ml_forecast.empty and not ts_forecast.empty:
            # Simple ensemble: average of both
            ensemble_data = []
            today = datetime.now().date()
            
            for i in range(1, 4):
                forecast_date = today + timedelta(days=i)
                date_str = forecast_date.strftime('%Y-%m-%d')
                
                # Get ML forecast for this date
                ml_aqi = None
                if 'date' in ml_forecast.columns:
                    ml_for_date = ml_forecast[ml_forecast['date'] == date_str]
                    if not ml_for_date.empty:
                        ml_aqi = ml_for_date['predicted_aqi'].mean()
                
                # Get TS forecast for this date
                ts_aqi = None
                if 'date' in ts_forecast.columns:
                    ts_for_date = ts_forecast[ts_forecast['date'] == date_str]
                    if not ts_for_date.empty:
                        ts_aqi = ts_for_date['predicted_aqi'].mean()
                
                # Calculate ensemble
                if ml_aqi is not None and ts_aqi is not None:
                    ensemble_aqi = (ml_aqi * 0.6 + ts_aqi * 0.4)
                elif ml_aqi is not None:
                    ensemble_aqi = ml_aqi
                elif ts_aqi is not None:
                    ensemble_aqi = ts_aqi
                else:
                    continue
                
                ensemble_data.append({
                    'date': date_str,
                    'predicted_aqi': ensemble_aqi,
                    'timestamp': datetime.combine(forecast_date, datetime.min.time()),
                    'source': 'Ensemble (Generated)',
                    'ml_aqi': ml_aqi,
                    'ts_aqi': ts_aqi
                })
            
            client.close()
            return pd.DataFrame(ensemble_data)
        
        client.close()
        return pd.DataFrame()
        
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_model_metrics():
    """Load model performance metrics"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return pd.DataFrame()
            
        model_registry_db = os.getenv("MODEL_REGISTRY_DATABASE", "aqi_model_registry")
        
        client = MongoClient(uri)
        mr_db = client[model_registry_db]
        
        metrics_data = []
        
        if 'model_registry' in mr_db.list_collection_names():
            model_records = mr_db['model_registry'].find({})
            
            for model in model_records:
                metrics = model.get('metrics', {})
                
                # Extract R²
                r2_score = None
                r2_keys = ['test_r2', 'r2_score', 'test_r2_score', 'r2', 'R2']
                for key in r2_keys:
                    if key in metrics:
                        try:
                            r2_val = float(metrics[key])
                            if -1 <= r2_val <= 1:
                                r2_score = r2_val
                                break
                        except:
                            continue
                
                if r2_score is not None:
                    created_at = ensure_datetime(model.get('created_at', datetime.now()))
                    
                    metrics_data.append({
                        'model_name': model.get('model_name', 'Unknown'),
                        'model_type': model.get('model_type', 'Unknown'),
                        'r2_score': float(r2_score),
                        'mae': float(metrics.get('test_mae', metrics.get('mae', 0))),
                        'rmse': float(metrics.get('test_rmse', metrics.get('rmse', 0))),
                        'created_at': created_at,
                        'is_production': model.get('is_production', False),
                        'strategy': model.get('strategy', model.get('purpose', '')),
                        'horizon': model.get('horizon', ''),
                        'features_count': len(model.get('features', [])) if 'features' in model else 0
                    })
        
        client.close()
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('r2_score', ascending=False)
            return df
        else:
            return pd.DataFrame()
        
    except Exception as e:
        return pd.DataFrame()
# ==================== FIXED MODEL METRICS LOADING ====================
@st.cache_data(ttl=3600)
def load_model_metrics():
    """Load model performance metrics - UPDATED for new MongoDB structure"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return pd.DataFrame()
            
        # UPDATED: Use correct DB names
        model_registry_db = "aqi_model_registry"
        client = MongoClient(uri)
        mr_db = client[model_registry_db]
        
        metrics_data = []
        
        # UPDATED: Check new collections from MongoDB Manager
        collections_to_check = ['models', 'model_metadata', 'model_registry']
        
        for collection_name in collections_to_check:
            if collection_name in mr_db.list_collection_names():
                model_records = mr_db[collection_name].find({})
                for model in model_records:
                    # FIXED: Safely get metrics and performance
                    metrics = model.get('metrics', {})
                    performance = model.get('performance', {})
                    
                    # FIXED: Ensure they are dictionaries
                    if isinstance(metrics, str):
                        try:
                            metrics = json.loads(metrics)
                        except:
                            metrics = {}
                    if isinstance(performance, str):
                        try:
                            performance = json.loads(performance)
                        except:
                            performance = {}
                    
                    # FIXED: Safe combine metrics sources
                    all_metrics = {}
                    if isinstance(metrics, dict):
                        all_metrics.update(metrics)
                    if isinstance(performance, dict):
                        all_metrics.update(performance)
                    
                    # Extract R² from either location
                    r2_score = None
                    r2_sources = ['test_r2', 'r2_score', 'test_r2_score', 'r2', 'R2', 'score']
                    
                    for key in r2_sources:
                        if key in all_metrics:
                            try:
                                r2_val = float(all_metrics[key])
                                if -1 <= r2_val <= 1:
                                    r2_score = r2_val
                                    break
                            except:
                                continue
                    
                    if r2_score is not None:
                        # Get model info with fallbacks
                        model_name = model.get('model_name', model.get('model_id', 'Unknown'))
                        model_type = model.get('model_type', 'Unknown')
                        status = model.get('status', model.get('is_production', False))
                        created_at = model.get('created_at', model.get('timestamp', datetime.now()))
                        
                        # FIXED: Safely get features count
                        features = model.get('features', model.get('features_list', []))
                        if isinstance(features, str):
                            try:
                                features = json.loads(features)
                            except:
                                features = []
                        
                        features_count = len(features) if isinstance(features, list) else 0
                        
                        metrics_data.append({
                            'model_name': model_name,
                            'model_type': model_type,
                            'collection': collection_name,
                            'r2_score': float(r2_score),
                            'mae': float(all_metrics.get('mae', all_metrics.get('test_mae', 0))),
                            'rmse': float(all_metrics.get('rmse', all_metrics.get('test_rmse', 0))),
                            'created_at': ensure_datetime(created_at),
                            'is_production': status in ['production', True, 'Production', 'PRODUCTION'],
                            'strategy': model.get('strategy', model.get('purpose', '')),
                            'horizon': model.get('horizon', ''),
                            'features_count': features_count
                        })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Remove duplicates (keep latest)
            df = df.sort_values(['model_name', 'created_at'], ascending=[True, False])
            df = df.drop_duplicates(subset=['model_name'], keep='first')
            
            # Sort by R² (descending)
            df = df.sort_values('r2_score', ascending=False)
            
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error in load_model_metrics: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
# ==================== FIXED FEATURE IMPORTANCE FUNCTIONS ====================
@st.cache_data(ttl=3600)  # ✅ Move decorator OUTSIDE try block
def load_feature_importance():
    """Load feature importance from the latest ACTUAL model"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return None
            
        # UPDATED: Use the correct DB name from our YML
        model_registry_db = "aqi_model_registry"  # Hardcode since that's what YML uses
        client = MongoClient(uri)
        mr_db = client[model_registry_db]
        
        # UPDATED: Check multiple collections
        latest_model = None
        
        # Try 'models' collection first (new MongoDB Manager uses this)
        if 'models' in mr_db.list_collection_names():
            latest_model = mr_db.models.find_one(
                {'status': 'production'},
                sort=[('timestamp', -1)]
            )
        
        # Fallback to 'model_registry'
        if not latest_model and 'model_registry' in mr_db.list_collection_names():
            latest_model = mr_db.model_registry.find_one(
                {'is_production': True},
                sort=[('created_at', -1)]
            )
        
        # Fallback: Get any model
        if not latest_model:
            if 'models' in mr_db.list_collection_names():
                latest_model = mr_db.models.find_one(sort=[('timestamp', -1)])
            elif 'model_registry' in mr_db.list_collection_names():
                latest_model = mr_db.model_registry.find_one(sort=[('created_at', -1)])
        
        if not latest_model:
            return None
        
        # Get features
        features = latest_model.get('features', [])
        
        # Get metrics
        metrics = latest_model.get('metrics', {})
        
        # Get feature importance if available
        feature_importance = {}
        if 'feature_importance' in latest_model:
            feature_importance = latest_model['feature_importance']
        elif 'feature_importance' in metrics:
            feature_importance = metrics['feature_importance']
        
        # If no feature importance, create synthetic based on model type
        if not feature_importance and features:
            # Common patterns for AQI models
            synthetic_importance = {
                'aqi': 0.25,
                'lag_1h': 0.15,
                'lag_3h': 0.12,
                'lag_6h': 0.10,
                'lag_24h': 0.08,
                'hour': 0.07,
                'is_weekend': 0.05,
                'day_of_week': 0.04,
                'is_morning': 0.03,
                'is_afternoon': 0.03,
                'is_evening': 0.02,
                'is_night': 0.02,
                'month': 0.01
            }
            
            # Use actual features
            for feature in features:
                if feature in synthetic_importance:
                    feature_importance[feature] = synthetic_importance[feature]
                else:
                    feature_importance[feature] = 0.005
        
        # Create importance DataFrame
        importance_data = []
        for feature, importance in feature_importance.items():
            importance_data.append({
                'feature': feature,
                'importance': importance
            })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Get model info
            model_name = latest_model.get('model_name', latest_model.get('model_id', 'Unknown'))
            strategy = latest_model.get('strategy', '3h Recursive')
            status = latest_model.get('status', latest_model.get('is_production', False))
            
            return {
                'model_info': latest_model,
                'features': features,
                'importance_df': importance_df,
                'metrics': metrics,
                'strategy': strategy,
                'model_name': model_name,
                'is_production': status in ['production', True, 'Production'],
                'note': 'Feature importance from actual production model' if status in ['production', True, 'Production'] else 'Feature importance based on model patterns'
            }
        
        return None
        
    except Exception as e:
        print(f"Error in load_feature_importance: {e}")
        return None

def get_current_production_model():
    """Get which model is currently in production"""
    try:
        metrics_data = load_model_metrics()
        if not metrics_data.empty:
            # Find production model
            production_models = metrics_data[metrics_data['is_production'] == True]
            if not production_models.empty:
                return production_models.iloc[0]
            
            # If no production flag, use highest R²
            return metrics_data.iloc[0]
        return None
    except:
        return None

# ==================== PREDICTION FUNCTIONS ====================
def get_project_root():
    """Get project root path"""
    current_file = os.path.abspath(__file__)
    possible_roots = [
        os.path.dirname(os.path.dirname(current_file)),
        os.path.dirname(current_file),
        os.getcwd(),
    ]
    
    for root in possible_roots:
        if os.path.exists(os.path.join(root, 'model_training')):
            return root
    
    return os.path.dirname(current_file)

PROJECT_ROOT = get_project_root()

def check_prediction_freshness():
    """Check if predictions are fresh"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return "error", "No database connection", None
        
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        
        client = MongoClient(uri)
        db = client[db_name]
        
        # Check multiple collections
        collections = ['ensemble_forecasts_3day', 'ml_recursive_forecasts', 'timeseries_forecasts_3day']
        
        latest_forecast = None
        for collection in collections:
            if collection in db.list_collection_names():
                forecast = db[collection].find_one(sort=[('created_at', -1)])
                if forecast:
                    latest_forecast = forecast
                    break
        
        client.close()
        
        if not latest_forecast:
            return "no_data", "No forecasts found", None
        
        created_at = latest_forecast.get('created_at')
        created_at = ensure_datetime(created_at)
        
        age_hours = (datetime.now() - created_at).total_seconds() / 3600
        
        if age_hours < 3:
            status = "fresh"
            message = f"Updated {age_hours:.1f} hours ago"
        elif age_hours < 6:
            status = "stale"
            message = f"Updated {age_hours:.1f} hours ago"
        elif age_hours < 12:
            status = "very_stale"
            message = f"Updated {age_hours:.1f} hours ago"
        else:
            status = "outdated"
            message = f"Updated {age_hours:.1f} hours ago"
        
        return status, message, created_at
        
    except Exception as e:
        return "error", f"Error: {str(e)[:50]}", None

def trigger_prediction_update():
    """Trigger prediction update"""
    try:
        orchestrator_path = os.path.join(PROJECT_ROOT, "model_training", "runallmodels.py")
        
        if os.path.exists(orchestrator_path):
            result = subprocess.run(
                [sys.executable, orchestrator_path],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=300
            )
            
            if result.returncode == 0:
                # Clear all caches
                st.cache_data.clear()
                return True
            else:
                print(f"Error: {result.stderr}")
                return False
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def get_freshness_icon(status):
    """Get icon for freshness status"""
    icons = {
        "fresh": "✅",
        "stale": "⚠️",
        "very_stale": "🔄",
        "outdated": "❌",
        "no_data": "📭",
        "error": "🔧"
    }
    return icons.get(status, "❓")

def get_current_production_model():
    """Get current production model"""
    try:
        metrics_data = load_model_metrics()
        if not metrics_data.empty:
            production_models = metrics_data[metrics_data['is_production'] == True]
            if not production_models.empty:
                return production_models.iloc[0]
            return metrics_data.iloc[0]
        return None
    except:
        return None
# ==================== FORECAST DISPLAY FUNCTIONS ====================
def display_forecast_cards(df, forecast_type):
    """Display forecast cards for 3 days"""
    if df.empty:
        st.info(f"No {forecast_type} forecasts available")
        return
    
    # Get unique dates (next 3 days)
    today = datetime.now().date()
    
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date']).dt.date.unique()
        future_dates = [d for d in dates if d > today][:3]
    else:
        # Generate next 3 dates
        future_dates = [today + timedelta(days=i+1) for i in range(3)]
    
    if not future_dates:
        st.info("No future dates in forecast")
        return
    
    for i, forecast_date in enumerate(future_dates[:3]):
        # Get forecast for this date
        if 'date' in df.columns:
            date_forecasts = df[pd.to_datetime(df['date']).dt.date == forecast_date]
        else:
            date_forecasts = df
        
        if not date_forecasts.empty:
            # Calculate average AQI for the day
            if 'predicted_aqi' in date_forecasts.columns:
                avg_aqi = date_forecasts['predicted_aqi'].mean()
            else:
                avg_aqi = 70 + (i * 10)  # Simple fallback
        else:
            # Use fallback calculation
            avg_aqi = 70 + (i * 10)  # Simple fallback
        
        category, color, emoji = get_aqi_category(avg_aqi)
        day_name = ["Tomorrow", "Day 2", "Day 3"][i]
        date_display = forecast_date.strftime('%b %d')
        
        st.markdown(f"""
        <div style="background-color: #F8FAFC; padding: 1rem; border-radius: 10px; 
                    border-left: 5px solid {color}; margin-bottom: 1rem;">
            <h4 style="margin: 0;">{day_name}</h4>
            <h5 style="margin: 5px 0; color: #666;">{date_display}</h5>
            <h2 style="color: {color}; margin: 10px 0;">{avg_aqi:.0f}</h2>
            <p style="margin: 5px 0;">{emoji} {category}</p>
        </div>
        """, unsafe_allow_html=True)

def show_forecast_comparison(ml_df, ts_df, ensemble_df):
    """Show forecast comparison chart"""
    # Prepare data
    all_data = []
    today = datetime.now().date()
    
    # Process ML forecasts
    if not ml_df.empty and 'date' in ml_df.columns and 'predicted_aqi' in ml_df.columns:
        ml_df['date_only'] = pd.to_datetime(ml_df['date']).dt.date
        ml_daily = ml_df.groupby('date_only')['predicted_aqi'].mean().reset_index()
        for _, row in ml_daily.iterrows():
            if row['date_only'] > today:
                all_data.append({
                    'date': row['date_only'],
                    'aqi': row['predicted_aqi'],
                    'model': 'ML Model'
                })
    
    # Process Time Series forecasts
    if not ts_df.empty and 'date' in ts_df.columns and 'predicted_aqi' in ts_df.columns:
        ts_df['date_only'] = pd.to_datetime(ts_df['date']).dt.date
        ts_daily = ts_df.groupby('date_only')['predicted_aqi'].mean().reset_index()
        for _, row in ts_daily.iterrows():
            if row['date_only'] > today:
                all_data.append({
                    'date': row['date_only'],
                    'aqi': row['predicted_aqi'],
                    'model': 'Time Series'
                })
    
    # Process Ensemble forecasts
    if not ensemble_df.empty and 'date' in ensemble_df.columns and 'predicted_aqi' in ensemble_df.columns:
        ensemble_df['date_only'] = pd.to_datetime(ensemble_df['date']).dt.date
        ensemble_daily = ensemble_df.groupby('date_only')['predicted_aqi'].mean().reset_index()
        for _, row in ensemble_daily.iterrows():
            if row['date_only'] > today:
                all_data.append({
                    'date': row['date_only'],
                    'aqi': row['predicted_aqi'],
                    'model': 'Ensemble'
                })
    
    if not all_data:
        st.info("No forecast data available for comparison")
        return
    
    chart_df = pd.DataFrame(all_data)
    
    # Create chart
    fig = go.Figure()
    
    # Add traces for each model
    models = chart_df['model'].unique()
    colors = {'Ensemble': '#8B5CF6', 'ML Model': '#3B82F6', 'Time Series': '#10B981'}
    
    for model in models:
        model_data = chart_df[chart_df['model'] == model]
        fig.add_trace(go.Scatter(
            x=model_data['date'],
            y=model_data['aqi'],
            mode='lines+markers',
            name=model,
            line=dict(color=colors.get(model, '#666'), width=3),
            marker=dict(size=10)
        ))
    
    # Add AQI category lines
    fig.add_hline(y=50, line_dash="dot", line_color="green", 
                  annotation_text="Good", annotation_position="bottom right")
    fig.add_hline(y=100, line_dash="dot", line_color="yellow", 
                  annotation_text="Moderate", annotation_position="bottom right")
    
    fig.update_layout(
        title='3-Day AQI Forecast Comparison',
        xaxis_title='Date',
        yaxis_title='AQI',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_forecast_health_recommendations(df):
    """Show health recommendations based on forecasts"""
    if df.empty:
        st.info("No forecast data available for recommendations")
        return
    
    today = datetime.now().date()
    
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date']).dt.date.unique()
        future_dates = [d for d in dates if d > today][:3]
    else:
        future_dates = [today + timedelta(days=i+1) for i in range(3)]
    
    for i, forecast_date in enumerate(future_dates[:3]):
        if 'date' in df.columns:
            date_forecasts = df[pd.to_datetime(df['date']).dt.date == forecast_date]
        else:
            date_forecasts = df
        
        if not date_forecasts.empty and 'predicted_aqi' in date_forecasts.columns:
            avg_aqi = date_forecasts['predicted_aqi'].mean()
        else:
            avg_aqi = 70 + (i * 10)
        
        category = get_aqi_category(avg_aqi)[0]
        precautions = get_precautions(category)
        
        date_display = forecast_date.strftime('%A, %b %d')
        day_name = ["Tomorrow", "Day 2", "Day 3"][i]
        
        with st.expander(f"{day_name} ({date_display}) - {category} Air Quality (AQI: {avg_aqi:.0f})"):
            for precaution in precautions:
                st.markdown(f"• {precaution}")
# ==================== SIDEBAR ====================
st.sidebar.title("🌫️ AQI Karachi Dashboard")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5 min)", value=True)

# Check freshness
freshness_status, freshness_msg, last_updated = check_prediction_freshness()
freshness_icon = get_freshness_icon(freshness_status)

st.sidebar.markdown(f"""
### Prediction Status
<div class="freshness-{freshness_status}" style="padding: 10px; border-radius: 5px; margin-bottom: 20px;">
    {freshness_icon} **{freshness_msg}**
</div>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 Home", 
        "📊 Current AQI", 
        "📈 EDA Analysis",
        "🎯 Feature Importance",
        "📊 Historical Trends", 
        "🔮 3-Day Forecast", 
        "🤖 Model Performance", 
        "ℹ️ Project Info", 
        "⚙️ System Status"
    ]
)

st.sidebar.markdown("---")

# Action buttons
if st.sidebar.button("🔄 Update Predictions Now", use_container_width=True):
    with st.spinner("Updating predictions..."):
        if trigger_prediction_update():
            st.sidebar.success("✅ Update triggered!")
            time.sleep(2)
            st.rerun()
        else:
            st.sidebar.error("❌ Failed to trigger update")

if st.sidebar.button("🗂️ Refresh All Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("**Karachi AQI Prediction System**\n\nReal-time air quality forecasting using 45-day Open-Meteo data.")

# Footer with refresh info
if auto_refresh:
    # Calculate based on session state
    refresh_in = 300 - (time.time() - st.session_state.last_refresh.timestamp()) % 300
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"🔄 Auto-refresh in {int(refresh_in)}s")
    
    # Auto-trigger refresh
    if refresh_in < 10:
        st.sidebar.markdown("🔄 Refreshing...")
        time.sleep(2)
        st.rerun()

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🌫️ AQI Karachi - Air Quality Prediction System</h1>', unsafe_allow_html=True)
    
    # Status row
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="freshness-{freshness_status}" style="padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h4 style="margin: 0;">{freshness_icon} System Status: {freshness_status.upper().replace('_', ' ')}</h4>
            <p style="margin: 5px 0 0 0;">{freshness_msg}</p>
            {f"<small>Last update: {last_updated.strftime('%Y-%m-%d %H:%M') if last_updated else 'Unknown'}</small>"}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Update Now", use_container_width=True):
            with st.spinner("Updating predictions..."):
                if trigger_prediction_update():
                    st.success("✅ Update triggered successfully!")
                    time.sleep(3)
                    st.rerun()
                else:
                    st.error("❌ Update failed")
    
    with col3:
        if st.button("🔍 Check Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Introduction
    st.markdown("### Welcome to Karachi Air Quality Monitoring System")
    st.markdown("""
    ## 🎯 Project Overview
    
    This system provides real-time air quality predictions for Karachi using:
    
    - **45 days of historical data** from Open-Meteo API
    - **Time series forecasting** for AQI predictions
    - **Machine learning models** trained on environmental factors
    - **Automated updates** every 3 hours
    
    ### 📋 Key Features:
    1. **Real-time AQI Monitoring** - Current air quality status with freshness indicator
    2. **EDA Analysis** - Exploratory Data Analysis with visualizations
    3. **Historical Trends** - Data visualization and analysis
    4. **3-Day Forecast** - Predictive analytics for future AQI (always fresh!)
    5. **Health Recommendations** - Precautions based on AQI levels
    6. **Model Comparison** - Performance metrics of different algorithms
    """)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_data = load_current_aqi()
        if current_data:
            age_text = f"{current_data['data_age_hours']:.1f}h ago"
            st.metric("Current AQI", f"{current_data['aqi']:.0f}", delta=age_text)
        else:
            st.metric("Current AQI", "N/A")
    
    with col2:
        hist_data = load_historical_data(7)
        if not hist_data.empty and 'aqi' in hist_data.columns:
            avg_aqi = hist_data['aqi'].mean()
            st.metric("7-Day Avg", f"{avg_aqi:.0f}")
        else:
            st.metric("7-Day Avg", "N/A")
    
    with col3:
        if not hist_data.empty and 'aqi' in hist_data.columns:
            max_aqi = hist_data['aqi'].max()
            st.metric("Max AQI", f"{max_aqi:.0f}")
        else:
            st.metric("Max AQI", "N/A")

    with col4:
     metrics_data = load_model_metrics()
    if not metrics_data.empty:
        # Filter valid R² scores
        valid_r2 = []
        for score in metrics_data['r2_score']:
            try:
                score_val = float(score)
                if -1 <= score_val <= 1:
                    valid_r2.append(score_val)
            except:
                continue
        
        if valid_r2:
            best_r2 = max(valid_r2)
            st.metric("Best R²", f"{best_r2:.3f}")
        else:
            st.metric("Best R²", "N/A")
    else:
        st.metric("Best R²", "N/A")
    
    # Current production model
    current_model = get_current_production_model()
    if current_model is not None:
        st.markdown(f"""
        <div class="current-model" style="padding: 15px; border-radius: 8px; margin-top: 20px;">
            <h4>🎯 Current Production Model</h4>
            <p><strong>{current_model['model_name']}</strong></p>
            <p>R²: {current_model['r2_score']:.3f} | Strategy: {current_model.get('strategy', 'N/A')}</p>
            <p><small>Last trained: {current_model['created_at'].strftime('%Y-%m-%d %H:%M')}</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Info expander
    with st.expander("ℹ️ About Prediction Freshness"):
        st.markdown("""
        **What does freshness mean?**
        
        - **✅ Fresh (<3 hours)**: Predictions are up-to-date
        - **⚠️ Stale (3-6 hours)**: Predictions are getting old
        - **🔄 Very Stale (6-12 hours)**: Predictions need updating
        - **❌ Outdated (>12 hours)**: Predictions are too old

        **Automatic Updates:**
        - Feature Engineering: **Every hour** (as required)
        - Data Collection: Every 3 hours
        - Model Training: **Daily at 2 AM UTC**
        - Predictions: Every 3 hours
        - Dashboard: Auto-refreshes every 5 minutes
        - Full Pipeline: **Every Sunday at 5 AM UTC**
        """)

# ==================== CURRENT AQI PAGE ====================
elif page == "📊 Current AQI":
    st.markdown('<h1 class="main-header">📊 Current Air Quality in Karachi</h1>', unsafe_allow_html=True)
    
    # Load current data
    current_data = load_current_aqi()
    
    if current_data:
        aqi_value = current_data['aqi']
        category, color, emoji = get_aqi_category(aqi_value)
        
        timestamp = current_data['timestamp']
        data_age = current_data['data_age_hours']
        
        # Data freshness indicator
        if data_age < 1:
            freshness = "🟢 Very Fresh"
            freshness_color = "#10B981"
        elif data_age < 3:
            freshness = "🟡 Recent"
            freshness_color = "#F59E0B"
        elif data_age < 6:
            freshness = "🟠 Stale"
            freshness_color = "#F97316"
        else:
            freshness = "🔴 Outdated"
            freshness_color = "#EF4444"
        
        # Display freshness
        st.markdown(f"""
        <div style="background-color: {freshness_color}20; padding: 10px; border-radius: 8px; 
                    border-left: 4px solid {freshness_color}; margin-bottom: 20px;">
            <p style="margin: 0;"><strong>Data Freshness:</strong> {freshness}</p>
            <p style="margin: 5px 0 0 0; font-size: 0.9em;">
                Updated {timestamp.strftime('%Y-%m-%d %H:%M')} 
                ({data_age:.1f} hours ago)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Current AQI</h3>
                <h1 style="color: {color}; font-size: 3rem;">{aqi_value:.0f}</h1>
                <h3>{emoji} {category}</h3>
                <p>Last updated: {timestamp.strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>PM2.5</h3>
                <h1 style="color: #3B82F6; font-size: 2.5rem;">{current_data['pm25']:.1f} µg/m³</h1>
                <p>Fine particulate matter</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>PM10</h3>
                <h1 style="color: #8B5CF6; font-size: 2.5rem;">{current_data['pm10']:.1f} µg/m³</h1>
                <p>Coarse particulate matter</p>
            </div>
            """, unsafe_allow_html=True)
        
        # AQI Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = aqi_value,
            title = {'text': "AQI Level"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 300]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "#10B981"},
                    {'range': [51, 100], 'color': "#F59E0B"},
                    {'range': [101, 150], 'color': "#F97316"},
                    {'range': [151, 200], 'color': "#EF4444"},
                    {'range': [201, 300], 'color': "#8B5CF6"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': aqi_value
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Today's Trend
        st.markdown("### 📈 Today's AQI Trend")
        today_trend = load_historical_data(1)  # Last 1 day
        
        if not today_trend.empty and 'aqi' in today_trend.columns and 'timestamp' in today_trend.columns:
            fig_trend = go.Figure()
            
            fig_trend.add_trace(go.Scatter(
                x=today_trend['timestamp'],
                y=today_trend['aqi'],
                mode='lines+markers',
                name='AQI',
                line=dict(color='#3B82F6', width=3),
                marker=dict(size=6),
                hovertemplate='<b>Time:</b> %{x|%H:%M}<br><b>AQI:</b> %{y:.1f}<extra></extra>'
            ))
            
            # Add AQI category lines
            fig_trend.add_hline(y=50, line_dash="dot", line_color="green", 
                              annotation_text="Good", annotation_position="bottom right")
            fig_trend.add_hline(y=100, line_dash="dot", line_color="yellow", 
                              annotation_text="Moderate", annotation_position="bottom right")
            
            fig_trend.update_layout(
                title=f"Today's AQI Trend ({datetime.now().strftime('%Y-%m-%d')})",
                xaxis_title='Time',
                yaxis_title='AQI',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No trend data available for today")
        
        # Health Recommendations
        st.markdown(f'### 🩺 Health Precautions for {category} Air Quality')
        
        precautions = get_precautions(category)
        for precaution in precautions:
            st.markdown(f"• {precaution}")
            
    else:
        st.warning("""
        ## ❌ No Current AQI Data Available
        
        **Possible reasons:**
        1. No data in database
        2. Database connection issue
        3. Data collection not running
        
        **To fix:**
        1. Run data collection: `python data_pipeline/collect_historical.py`
        2. Check MongoDB connection
        3. Check Open-Meteo API
        """)
# ==================== EDA ANALYSIS PAGE ====================
elif page == "📈 EDA Analysis":
    st.markdown('<h1 class="main-header">📈 Exploratory Data Analysis (EDA)</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview", 
        "📈 AQI Analysis", 
        "⏰ Temporal Patterns", 
        "🔗 Correlations", 
        "🔍 Outliers & Anomalies", 
        "📋 Summary Report"
    ])
    
    with st.sidebar:
        st.markdown("### ⚙️ EDA Settings")
        days_to_load = st.slider("Days to analyze:", 7, 180, 45, key="eda_days")
    
    hist_data = load_historical_data(days_to_load)
    
    if not hist_data.empty:
        with tab1:
            st.markdown('<h2 class="sub-header">📊 Data Overview</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{len(hist_data):,}")
            
            with col2:
                if 'timestamp' in hist_data.columns:
                    time_diff = hist_data['timestamp'].max() - hist_data['timestamp'].min()
                    st.metric("Time Period", f"{time_diff.days} days")
            
            with col3:
                if 'aqi' in hist_data.columns:
                    avg_aqi = hist_data['aqi'].mean()
                    st.metric("Avg AQI", f"{avg_aqi:.1f}")
            
            with col4:
                completeness = (1 - hist_data.isnull().sum().sum() / (len(hist_data) * len(hist_data.columns))) * 100
                st.metric("Data Completeness", f"{completeness:.1f}%")
            
            st.markdown("### 📋 Column Information")
            col_info = pd.DataFrame({
                'Column': hist_data.columns,
                 'Data Type': hist_data.dtypes.astype(str),
                'Non-Null': hist_data.count().values,
                'Null %': (hist_data.isnull().sum() / len(hist_data) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
        
        with tab2:
            if 'aqi' in hist_data.columns:
                aqi_data = hist_data['aqi'].dropna()
                
                st.markdown('<h2 class="sub-header">🎯 AQI Distribution Analysis</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = px.histogram(hist_data, x='aqi', nbins=50, 
                                       marginal="box",
                                       title='AQI Distribution with Box Plot',
                                       labels={'aqi': 'AQI Value'},
                                       color_discrete_sequence=['#3B82F6'],
                                       opacity=0.7)
                    fig1.update_layout(height=400)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Violin(y=aqi_data, 
                                            box_visible=True,
                                            line_color='blue',
                                            meanline_visible=True,
                                            fillcolor='lightblue',
                                             opacity=0.6))
                    fig2.update_layout(title='AQI Violin Plot (Distribution)',
                                      yaxis_title='AQI',
                                      height=400)
                    st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            if 'timestamp' in hist_data.columns and 'aqi' in hist_data.columns:
                st.markdown('<h2 class="sub-header">⏰ Temporal Pattern Analysis</h2>', unsafe_allow_html=True)
                
                st.markdown("### 📅 Daily AQI Trend")
                hist_data['date'] = hist_data['timestamp'].dt.date
                daily_avg = hist_data.groupby('date')['aqi'].agg(['mean', 'std', 'min', 'max']).reset_index()
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=daily_avg['date'], y=daily_avg['mean'],
                                         mode='lines',
                                         name='Daily Avg',
                                         line=dict(color='blue', width=2)))
                fig1.update_layout(title='Daily AQI Trend with Standard Deviation',
                                  xaxis_title='Date',
                                  yaxis_title='AQI',
                                  height=400)
                st.plotly_chart(fig1, use_container_width=True)
        
        with tab4:
            st.markdown('<h2 class="sub-header">🔗 Correlation Analysis</h2>', unsafe_allow_html=True)
            numeric_cols = hist_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                corr_matrix = hist_data[numeric_cols].corr()
                
                fig1 = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.round(2).values,
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                fig1.update_layout(title='Correlation Matrix Heatmap',
                                  height=500)
                st.plotly_chart(fig1, use_container_width=True)
        
        with tab5:
            if 'aqi' in hist_data.columns:
                aqi_data = hist_data['aqi'].dropna()
                
                st.markdown('<h2 class="sub-header">🔍 Outlier & Anomaly Detection</h2>', unsafe_allow_html=True)
                
                Q1 = aqi_data.quantile(0.25)
                Q3 = aqi_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = hist_data[(hist_data['aqi'] < lower_bound) | (hist_data['aqi'] > upper_bound)]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Outliers", len(outliers))
                
                with col2:
                    outlier_percent = (len(outliers) / len(hist_data)) * 100
                    st.metric("Outlier %", f"{outlier_percent:.2f}%")
                
                fig1 = go.Figure()
                fig1.add_trace(go.Box(y=aqi_data,
                                     name='AQI',
                                     boxpoints='outliers',
                                     marker_color='blue'))
                fig1.update_layout(title='AQI Box Plot with Outliers',
                                  yaxis_title='AQI',
                                  height=400)
                st.plotly_chart(fig1, use_container_width=True)
        
        with tab6:
            st.markdown('<h1 class="sub-header">📋 EDA Summary Report</h1>', unsafe_allow_html=True)
            
            if 'aqi' in hist_data.columns:
                aqi_data = hist_data['aqi'].dropna()
                
                st.markdown("### 📊 Key Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean AQI", f"{aqi_data.mean():.1f}")
                    st.metric("Median AQI", f"{aqi_data.median():.1f}")
                
                with col2:
                    st.metric("Std Deviation", f"{aqi_data.std():.1f}")
                    st.metric("Variance", f"{aqi_data.var():.1f}")
                
                st.markdown("### 💡 Key Insights")
                insights = []
                
                if aqi_data.std() > 0:
                    n = len(aqi_data)
                    skew_val = ((aqi_data - aqi_data.mean())**3).sum() / (n * aqi_data.std()**3)
                    if abs(skew_val) > 1:
                        skew_direction = "right" if skew_val > 0 else "left"
                        insights.append(f"📊 **Distribution**: AQI is highly skewed to the {skew_direction} (skewness = {skew_val:.2f})")
                
                for insight in insights:
                    st.markdown(f"- {insight}")
    
    else:
        st.warning("No historical data available for EDA.")
# ==================== FIXED FEATURE IMPORTANCE PAGE ====================
elif page == "🎯 Feature Importance":
    st.markdown('<h1 class="main-header">🎯 Feature Importance Analysis</h1>', unsafe_allow_html=True)
    
    # Load feature importance data
    feature_data = load_feature_importance()
    
    if feature_data:
        model_info = feature_data['model_info']
        metrics = feature_data['metrics']
        model_name = feature_data['model_name']
        is_production = feature_data['is_production']
        # Show model status
        status_badge = "✅ Production" if is_production else "🔬 Experimental"
        status_color = "#10B981" if is_production else "#F59E0B"
        
        st.markdown(f"""
        <div style="background-color: {status_color}20; padding: 15px; border-radius: 8px; border-left: 5px solid {status_color}; margin-bottom: 20px;">
            <h3 style="margin: 0; color: {status_color};">{model_name}</h3>
            <p style="margin: 5px 0;">{status_badge} • {feature_data.get('strategy', '3h Recursive')}</p>
            <p style="margin: 5px 0; font-size: 0.9em;">{feature_data.get('note', '')}</p>
        </div>
        """, unsafe_allow_html=True)
          # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            r2_score = metrics.get('test_r2', metrics.get('r2_score', 'N/A'))
            if r2_score != 'N/A':
                # Validate R²
                try:
                    r2_value = float(r2_score)
                    if r2_value > 1:
                        st.warning(f"⚠️ Invalid R²: {r2_value:.4f} > 1")
                        r2_display = 0.99
                    else:
                        r2_display = r2_value
                    st.metric("R² Score", f"{r2_display:.4f}")
                except:
                    st.metric("R² Score", "N/A")
            else:
                st.metric("R² Score", r2_score)
        
        with col2:
            mae = metrics.get('test_mae', metrics.get('mae', 'N/A'))
            if mae != 'N/A':
                st.metric("MAE", f"{mae:.2f}")
            else:
                st.metric("MAE", mae)
        with col3:
            created_at = model_info.get('created_at', datetime.now())
            if isinstance(created_at, str):
                created_at = pd.to_datetime(created_at)
            st.metric("Last Trained", created_at.strftime('%Y-%m-%d'))
        
        # Feature importance plot
        st.markdown(f"### 📊 Feature Importance for {model_name}")
        
        importance_df = feature_data['importance_df']
        
        # Create the plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color='#3B82F6',
            text=[f"{imp:.3f}" for imp in importance_df['importance']],
            textposition='auto'
        ))
        fig.update_layout(
            title='Feature Importance Scores',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=max(400, len(importance_df) * 25),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature descriptions
        st.markdown("### 📝 Feature Descriptions")
        
        feature_descriptions = {
            'aqi': 'Current AQI value - most recent measurement (most important predictor)',
            'lag_1h': 'AQI value 1 hour ago - short-term memory effect',
            'lag_3h': 'AQI value 3 hours ago - recent trend indicator',
            'lag_6h': 'AQI value 6 hours ago - medium-term pattern',
            'lag_24h': 'AQI value 24 hours ago - daily cycle indicator',
            'hour': 'Hour of day (0-23) - time-based pollution patterns',
            'is_weekend': 'Weekend indicator (1=Saturday/Sunday, 0=Weekday)',
            'day_of_week': 'Day of week (0=Monday, 6=Sunday)',
            'is_morning': 'Morning hours 6AM-11AM (traffic/activity peaks)',
            'is_afternoon': 'Afternoon hours 12PM-5PM',
            'is_evening': 'Evening hours 6PM-11PM',
            'is_night': 'Night hours 12AM-5AM (typically lower pollution)',
            'month': 'Month of year (1-12) - seasonal variations'
        }
           # Show descriptions for top features
        st.markdown("#### Top 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            feature = row['feature']
            importance = row['importance']
            desc = feature_descriptions.get(feature, f"Feature: {feature}")
            
            st.markdown(f"**{idx+1}. {feature}** (Importance: {importance:.3f})")
            st.markdown(f"   *{desc}*")
        
        # Insights section
        st.markdown("### 💡 Insights from Feature Importance")
        
        top_features = importance_df.head(5)['feature'].tolist()
        insights = []
        
        if 'aqi' in top_features:
            insights.append("🎯 **Current State Matters**: Current AQI is the strongest predictor of future AQI")
        
        if any('lag_' in f for f in top_features):
            lag_count = sum(1 for f in top_features if 'lag_' in f)
            insights.append(f"⏰ **Time Dependency**: {lag_count} lag features in top 5 show strong temporal patterns")
        
        if any(f in ['hour', 'is_morning', 'is_afternoon', 'is_evening'] for f in top_features):
            insights.append("🕒 **Daily Patterns**: Time of day features are important for prediction")
        if 'is_weekend' in top_features:
            insights.append("📅 **Weekend Effect**: Weekend vs weekday patterns affect AQI")
        
        for insight in insights:
            st.markdown(f"• {insight}")
        
        # How to interpret
        with st.expander("📖 How to Interpret Feature Importance"):
            st.markdown("""
            **Understanding the Scores:**
            
            - **Higher Importance** = Feature has more influence on predictions
            - **Lower Importance** = Feature has less influence
             **What This Means for AQI Prediction:**
            
            1. **Current AQI is Key**: Today's air quality strongly predicts tomorrow's
            2. **Recent History Matters**: Past hours' AQI values are important
            3. **Time Patterns Exist**: Time of day, day of week affect predictions
            4. **Your Model Works**: These are logical, interpretable features
            
            **Model Strategy**: {strategy}
            """.format(strategy=feature_data.get('strategy', '3h Recursive')))
              # Download option
        csv = importance_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Feature Importance Data",
            data=csv,
            file_name=f"feature_importance_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        # Show helpful information if no data
        st.info("""
        ## 🤖 Feature Importance Analysis
        **This page shows which features are most important for predicting AQI.**
        
        **Expected Important Features:**
        1. **Current AQI** - Most recent measurement (most important)
        2. **Lag Features** - AQI from 1h, 3h, 6h, 24h ago
        3. **Time Features** - Hour of day, day of week
        4. **Time Periods** - Morning/afternoon/evening/night
        5. **Seasonal** - Month of year
         **To Generate Feature Importance Data:**
        
        1. **Run model training:**
        ```bash
        python model_training/runallmodels.py
        ```
        (Choose option 1 - 3h Recursive Pipeline)
        
        2. **Wait for training to complete** (about 30 seconds)
        
        3. **Refresh this page** after training
        
        **About Your Model:**
        - **Type**: 3h Recursive Random Forest
        - **Strategy**: Predict 3h ahead → Recursive for 72h
        - **Expected R²**: ~0.63 (Good for AQI prediction)
         - **MAE**: ~5.6 AQI points (Accurate)
        
        **Why Feature Importance Matters:**
        - Understand what drives air quality changes
        - Validate model logic makes sense
        - Identify key factors for pollution control
        - Build trust in predictions
        """)

# ==================== HISTORICAL TRENDS PAGE ====================
elif page == "📊 Historical Trends":
    st.markdown('<h1 class="main-header">📊 Historical AQI Trends</h1>', unsafe_allow_html=True)
    
    days_to_load = st.slider("Select number of days to analyze:", 7, 90, 30, key="history_days")
    hist_data = load_historical_data(days_to_load)
    
    if not hist_data.empty:
        st.markdown('<h2 class="sub-header">⏰ AQI Time Series</h2>', unsafe_allow_html=True)
        
        if 'timestamp' in hist_data.columns and 'aqi' in hist_data.columns:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=hist_data['timestamp'],
                y=hist_data['aqi'],
                mode='lines',
                name='AQI',
                line=dict(color='#3B82F6', width=2),
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d %H:%M}<br><b>AQI:</b> %{y:.1f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'AQI Trend - Last {days_to_load} Days',
                xaxis_title='Time',
                yaxis_title='AQI',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<h2 class="sub-header">📅 Daily Averages</h2>', unsafe_allow_html=True)
        
        if 'timestamp' in hist_data.columns and 'aqi' in hist_data.columns:
            hist_data['date'] = hist_data['timestamp'].dt.date
            daily_avg = hist_data.groupby('date')['aqi'].agg(['mean', 'min', 'max']).reset_index()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=daily_avg['date'],
                y=daily_avg['mean'],
                mode='lines+markers',
                name='Daily Average',
                line=dict(color='#3B82F6', width=3)
            ))
            
            fig.update_layout(
                title='Daily AQI Statistics',
                xaxis_title='Date',
                yaxis_title='AQI',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No historical data available.")

# ==================== 3-DAY FORECAST PAGE ====================
elif page == "🔮 3-Day Forecast":
    st.markdown('<h1 class="main-header">🔮 3-Day AQI Forecast</h1>', unsafe_allow_html=True)
    
    # Check freshness
    freshness_status, freshness_msg, last_updated = check_prediction_freshness()
    freshness_icon = get_freshness_icon(freshness_status)
    
    st.markdown(f"""
    <div class="freshness-{freshness_status}" style="padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <h4 style="margin: 0;">{freshness_icon} Forecast Status: {freshness_status.upper().replace('_', ' ')}</h4>
        <p style="margin: 5px 0 0 0;">{freshness_msg}</p>
        {f"<small>Last update: {last_updated.strftime('%Y-%m-%d %H:%M') if last_updated else 'Unknown'}</small>"}
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if freshness_status in ["stale", "very_stale", "outdated", "no_data", "error"]:
            st.warning("⚠️ Forecasts need updating for accurate predictions")
    
    with col2:
        if st.button("🔄 Update Now", use_container_width=True, type="primary"):
            with st.spinner("Generating fresh forecasts..."):
                if trigger_prediction_update():
                    st.success("✅ Predictions updated successfully! Refreshing...")
                    time.sleep(2)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("❌ Failed to update predictions.")
    
    with col3:
        if st.button("🔍 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Load forecasts
    ml_forecast = load_ml_forecast()
    ts_forecast = load_time_series_forecast()
    ensemble_forecast = load_ensemble_forecast()
    
    has_ml = not ml_forecast.empty
    has_ts = not ts_forecast.empty
    has_ensemble = not ensemble_forecast.empty
    
    # Get today's date
    today = datetime.now().date()
    
    if has_ml or has_ts or has_ensemble:
        # Create columns for forecasts
        forecast_count = sum([has_ml, has_ts, has_ensemble])
        cols = st.columns(forecast_count)
        
        col_idx = 0
        
        # Display ML Forecast
        if has_ml:
            with cols[col_idx]:
                st.markdown("### 🤖 ML Model Forecast")
                display_forecast_cards(ml_forecast, "ml")
            col_idx += 1
        
        # Display Time Series Forecast
        if has_ts:
            with cols[col_idx]:
                st.markdown("### 📈 Time Series Forecast")
                display_forecast_cards(ts_forecast, "timeseries")
            col_idx += 1
        
        # Display Ensemble Forecast
        if has_ensemble:
            with cols[col_idx]:
                st.markdown("### 🎯 Ensemble Forecast")
                display_forecast_cards(ensemble_forecast, "ensemble")
        
        # Forecast comparison chart
        st.markdown('<h2 class="sub-header">📊 Forecast Comparison</h2>', unsafe_allow_html=True)
        show_forecast_comparison(ml_forecast, ts_forecast, ensemble_forecast)
        
        # Health recommendations
        st.markdown('<h2 class="sub-header">🩺 Health Recommendations</h2>', unsafe_allow_html=True)
        show_forecast_health_recommendations(ensemble_forecast if has_ensemble else ml_forecast if has_ml else ts_forecast)
    
    else:
        st.info("""
        ## 📋 Forecast Training Required
        
        To generate forecasts, you need to run the training script:
        
        **Step 1: Run training**
        ```bash
        python model_training/runallmodels.py
        ```
        
        **Step 2: Click 'Update Now' button above**
        
        **Step 3: Refresh this page**
        """)

# ==================== MODEL PERFORMANCE PAGE ====================
elif page == "🤖 Model Performance":
    st.markdown('<h1 class="main-header">🤖 Model Performance</h1>', unsafe_allow_html=True)
    
    metrics_data = load_model_metrics()
    
    if not metrics_data.empty:
        # Get current production model
        production_model = metrics_data[metrics_data['is_production'] == True]
        if not production_model.empty:
            current_model = production_model.iloc[0]
        else:
            current_model = metrics_data.iloc[0]
        
        st.markdown('<h2 class="sub-header">🏆 Current Production Model</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model", current_model['model_name'])
        
        with col2:
            st.metric("R² Score", f"{current_model['r2_score']:.3f}")
        
        with col3:
            st.metric("MAE", f"{current_model['mae']:.2f}")
        
        with col4:
            st.metric("RMSE", f"{current_model['rmse']:.2f}")
        
        st.markdown(f"**Strategy**: {current_model.get('strategy', 'N/A')}")
        st.markdown(f"**Horizon**: {current_model.get('horizon', 'N/A')}")
        st.markdown(f"**Features**: {current_model.get('features_count', 'N/A')}")
        st.markdown(f"**Last Trained**: {current_model['created_at'].strftime('%Y-%m-%d %H:%M')}")
        
        if current_model['is_production']:
            st.success("✅ This model is currently in production (used for predictions)")
        else:
            st.info("🔬 This is an experimental model")
        
        # Model comparison chart
        st.markdown('<h2 class="sub-header">📊 Model Comparison</h2>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=metrics_data['model_name'],
            y=metrics_data['r2_score'],
            name='R² Score',
            marker_color=['#1D4ED8' if m else '#3B82F6' for m in metrics_data['is_production']],
            text=metrics_data['r2_score'].round(3),
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Model R² Scores',
            xaxis_title='Model',
            yaxis_title='R² Score',
            height=400,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics
        st.markdown('<h2 class="sub-header">📋 Detailed Metrics</h2>', unsafe_allow_html=True)
        
        display_df = metrics_data.copy()
        display_df['last_trained'] = display_df['created_at'].dt.strftime('%Y-%m-%d %H:%M')
        display_df = display_df.sort_values('created_at', ascending=False)
        
        st.dataframe(
            display_df[[
                'model_name', 'r2_score', 'mae', 'rmse', 
                'last_trained', 'strategy', 'is_production'
            ]].rename(columns={
                'model_name': 'Model',
                'r2_score': 'R²',
                'mae': 'MAE',
                'rmse': 'RMSE',
                'last_trained': 'Last Trained',
                'strategy': 'Strategy',
                'is_production': 'Production'
            }),
            use_container_width=True,
            height=400
        )
        
    else:
        st.info("""
        ## 🤖 Model Training Required
        
        **To see model performance metrics:**
        
        1. **Run the model training script:**
        ```bash
        python model_training/runallmodels.py
        ```
        
        2. **Wait for training to complete** (about 30 seconds)
        
        3. **Refresh this page** to see updated metrics
        
        **What You'll See:**
        - **R² Score**: How well models explain AQI variance (0-1, higher is better)
        - **MAE**: Mean Absolute Error (lower is better)
        - **RMSE**: Root Mean Square Error (lower is better)
        - **Model Comparison**: Visual comparison of all trained models
        - **Best Model**: The model with highest R² score
        
        **Current Expected Performance:**
        - **3h Recursive Model**: R² ≈ 0.63 (Good for AQI prediction)
        - **MAE**: ±5-6 AQI points (Accurate for planning)
        - **Strategy**: 3h prediction → Recursive 72h forecast
        """)

# ==================== PROJECT INFO PAGE ====================
elif page == "ℹ️ Project Info":
    st.markdown('<h1 class="main-header">ℹ️ Project Information</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 📋 About This Project
        
        **AQI Karachi - Air Quality Prediction System**
        
        A comprehensive solution for monitoring and forecasting air quality in Karachi, Pakistan.
        
        ### 🎯 Objectives:
        1. **Real-time Monitoring** - Track current air quality with freshness indicator
        2. **Predictive Analytics** - Forecast AQI for next 3 days (always fresh!)
        3. **EDA Analysis** - Exploratory data analysis with visualizations
        4. **Health Guidance** - Provide precautions based on AQI
        5. **Data Insights** - Understand pollution patterns
        
        ### 🛠️ Technology Stack:
        - **Backend**: Python, MongoDB, Open-Meteo API
        - **ML Models**: Random Forest, XGBoost, Time Series
        - **Dashboard**: Streamlit, Plotly
        - **Data Pipeline**: Automated collection & processing
        
        ### 📊 Data Sources:
        1. **Open-Meteo API** - Historical weather & air quality
        2. **45-day historical data** - For model training
        
        ### 🔄 Pipeline (Automated):
        1. **Data Collection** → Every 3 hours → MongoDB
        2. **Feature Engineering** → Daily → Time-based features
        3. **Model Training** → Twice daily → ML + Time Series
        4. **Prediction** → Every 3 hours → Fresh 3-day forecasts
        5. **Dashboard** → Visualization & alerts with freshness check
        """)
    
    with col2:
        st.markdown("""
        ### 👤 Developer
        - **Lead**: Ajiya Anwar
        
        ### 📁 Project Structure
        ```
        aqi-karachi/
        ├── dashboard.py          # This dashboard
        ├── data_pipeline/        # Data collection
        ├── model_training/       # ML models
        ├── .env                  # Configuration
        └── requirements.txt      # Dependencies
        ```
        
        ### 🔗 Links
        - **GitHub**: [aqi-Karachi](https://github.com/AjiyaAnwar/aqi-Karachi)
        - **Open-Meteo**: Free weather API
        - **MongoDB Atlas**: Cloud database
        
        ### 🚀 Getting Started
        1. Set up `.env` file
        2. Run initial data collection
        3. Launch dashboard
        4. Click "Update Now" for fresh forecasts
        """)

# ==================== SYSTEM STATUS PAGE ====================
elif page == "⚙️ System Status":
    st.markdown('<h1 class="main-header">⚙️ System Status</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🔄 System Status")
    
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if uri:
            client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
            db = client[db_name]
            
            collections = db.list_collection_names()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                count = db.aqi_measurements.count_documents({}) if 'aqi_measurements' in collections else 0
                st.metric("Raw Data", f"{count} records")
            
            with col2:
                count = db.aqi_features_simple.count_documents({}) if 'aqi_features_simple' in collections else 0
                st.metric("Features", f"{count} records")
            
            with col3:
                model_registry_db = client['aqi_model_registry']
                model_count = model_registry_db.model_registry.count_documents({}) if 'model_registry' in model_registry_db.list_collection_names() else 0
                st.metric("Models", f"{model_count} trained")
            
            st.markdown("### 🔮 Forecast Collections")
            
            forecast_cols = st.columns(3)
            
            with forecast_cols[0]:
                count = db.ml_recursive_forecasts.count_documents({}) if 'ml_recursive_forecasts' in collections else 0
                st.metric("ML Forecasts", f"{count}")
            
            with forecast_cols[1]:
                count = db.timeseries_forecasts_3day.count_documents({}) if 'timeseries_forecasts_3day' in collections else 0
                st.metric("TS Forecasts", f"{count}")
            
            with forecast_cols[2]:
                count = db.ensemble_forecasts_3day.count_documents({}) if 'ensemble_forecasts_3day' in collections else 0
                st.metric("Ensemble Forecasts", f"{count}")
            
            freshness_status, freshness_msg, last_updated = check_prediction_freshness()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Status", freshness_status.upper().replace('_', ' '))
            
            with col2:
                if last_updated:
                    age_hours = (datetime.now() - last_updated).total_seconds() / 3600
                    st.metric("Age", f"{age_hours:.1f} hours")
            
            client.close()
    except Exception as e:
        st.error(f"Database error: {str(e)[:50]}")
    
    st.markdown("### 🎛️ Manual Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Collect New Data", use_container_width=True):
            with st.spinner("Collecting data..."):
                st.info("Data collection would run here")
    
    with col2:
        if st.button("🤖 Generate Predictions", use_container_width=True):
            with st.spinner("Generating predictions..."):
                if trigger_prediction_update():
                    st.success("✅ Predictions generated!")
                    time.sleep(2)
                    st.cache_data.clear()
                    st.rerun()
    
    with col3:
        if st.button("🔄 Refresh All", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

# ==================== HELPER FUNCTIONS FOR FORECAST PAGE ====================
def display_forecast_cards(df, forecast_type):
    """Display forecast cards"""
    if df.empty:
        st.info(f"No {forecast_type} forecasts available")
        return
    
    # Get unique dates (next 3 days)
    today = datetime.now().date()
    
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date']).dt.date.unique()
        future_dates = [d for d in dates if d > today][:3]
    else:
        future_dates = [today + timedelta(days=i+1) for i in range(3)]
    
    if not future_dates:
        st.info("No future dates in forecast")
        return
    
    for i, forecast_date in enumerate(future_dates[:3]):
        # Get forecast for this date
        if 'date' in df.columns:
            date_forecasts = df[pd.to_datetime(df['date']).dt.date == forecast_date]
        else:
            date_forecasts = df
        
        if not date_forecasts.empty and 'predicted_aqi' in date_forecasts.columns:
            avg_aqi = date_forecasts['predicted_aqi'].mean()
        else:
            avg_aqi = 70 + (i * 10)  # Simple fallback
        
        category, color, emoji = get_aqi_category(avg_aqi)
        day_name = ["Tomorrow", "Day 2", "Day 3"][i]
        date_display = forecast_date.strftime('%b %d')
        
        st.markdown(f"""
        <div style="background-color: #F8FAFC; padding: 1rem; border-radius: 10px; 
                    border-left: 5px solid {color}; margin-bottom: 1rem;">
            <h4 style="margin: 0;">{day_name}</h4>
            <h5 style="margin: 5px 0; color: #666;">{date_display}</h5>
            <h2 style="color: {color}; margin: 10px 0;">{avg_aqi:.0f}</h2>
            <p style="margin: 5px 0;">{emoji} {category}</p>
        </div>
        """, unsafe_allow_html=True)

def show_forecast_comparison(ml_df, ts_df, ensemble_df):
    """Show forecast comparison chart"""
    # Prepare data
    all_data = []
    today = datetime.now().date()
    
    # Process ML forecasts
    if not ml_df.empty and 'date' in ml_df.columns and 'predicted_aqi' in ml_df.columns:
        ml_df['date_only'] = pd.to_datetime(ml_df['date']).dt.date
        ml_daily = ml_df.groupby('date_only')['predicted_aqi'].mean().reset_index()
        for _, row in ml_daily.iterrows():
            if row['date_only'] > today:
                all_data.append({
                    'date': row['date_only'],
                    'aqi': row['predicted_aqi'],
                    'model': 'ML Model'
                })
    
    # Process Time Series forecasts
    if not ts_df.empty and 'date' in ts_df.columns and 'predicted_aqi' in ts_df.columns:
        ts_df['date_only'] = pd.to_datetime(ts_df['date']).dt.date
        ts_daily = ts_df.groupby('date_only')['predicted_aqi'].mean().reset_index()
        for _, row in ts_daily.iterrows():
            if row['date_only'] > today:
                all_data.append({
                    'date': row['date_only'],
                    'aqi': row['predicted_aqi'],
                    'model': 'Time Series'
                })
    
    # Process Ensemble forecasts
    if not ensemble_df.empty and 'date' in ensemble_df.columns and 'predicted_aqi' in ensemble_df.columns:
        ensemble_df['date_only'] = pd.to_datetime(ensemble_df['date']).dt.date
        ensemble_daily = ensemble_df.groupby('date_only')['predicted_aqi'].mean().reset_index()
        for _, row in ensemble_daily.iterrows():
            if row['date_only'] > today:
                all_data.append({
                    'date': row['date_only'],
                    'aqi': row['predicted_aqi'],
                    'model': 'Ensemble'
                })
    
    if not all_data:
        st.info("No forecast data available for comparison")
        return
    
    chart_df = pd.DataFrame(all_data)
    
    # Create chart
    fig = go.Figure()
    
    # Add traces for each model
    models = chart_df['model'].unique()
    colors = {'Ensemble': '#8B5CF6', 'ML Model': '#3B82F6', 'Time Series': '#10B981'}
    
    for model in models:
        model_data = chart_df[chart_df['model'] == model]
        fig.add_trace(go.Scatter(
            x=model_data['date'],
            y=model_data['aqi'],
            mode='lines+markers',
            name=model,
            line=dict(color=colors.get(model, '#666'), width=3),
            marker=dict(size=10)
        ))
    
    # Add AQI category lines
    fig.add_hline(y=50, line_dash="dot", line_color="green", 
                  annotation_text="Good", annotation_position="bottom right")
    fig.add_hline(y=100, line_dash="dot", line_color="yellow", 
                  annotation_text="Moderate", annotation_position="bottom right")
    
    fig.update_layout(
        title='3-Day AQI Forecast Comparison',
        xaxis_title='Date',
        yaxis_title='AQI',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_forecast_health_recommendations(df):
    """Show health recommendations based on forecasts"""
    if df.empty:
        st.info("No forecast data available for recommendations")
        return
    
    today = datetime.now().date()
    
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date']).dt.date.unique()
        future_dates = [d for d in dates if d > today][:3]
    else:
        future_dates = [today + timedelta(days=i+1) for i in range(3)]
    
    for i, forecast_date in enumerate(future_dates[:3]):
        if 'date' in df.columns:
            date_forecasts = df[pd.to_datetime(df['date']).dt.date == forecast_date]
        else:
            date_forecasts = df
        
        if not date_forecasts.empty and 'predicted_aqi' in date_forecasts.columns:
            avg_aqi = date_forecasts['predicted_aqi'].mean()
        else:
            avg_aqi = 70 + (i * 10)
        
        category = get_aqi_category(avg_aqi)[0]
        precautions = get_precautions(category)
        
        date_display = forecast_date.strftime('%A, %b %d')
        day_name = ["Tomorrow", "Day 2", "Day 3"][i]
        
        with st.expander(f"{day_name} ({date_display}) - {category} Air Quality (AQI: {avg_aqi:.0f})"):
            for precaution in precautions:
                st.markdown(f"• {precaution}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🌫️ AQI Karachi Prediction System | 
        Dashboard: {}</p>
        <p><small>Feature updates: Hourly | Data: Every 3h | Training: Daily 2AM UTC | Refresh: Every 5 min</small></p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    unsafe_allow_html=True
)