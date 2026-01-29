"""
📊 AQI-Karachi Dashboard: COMPLETE VERSION WITH FRESHNESS CHECK
Includes Full EDA + Feature Importance + 3-Day Forecasts + Freshness Indicator
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
from dotenv import load_dotenv
import warnings
import time
warnings.filterwarnings('ignore')

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if os.path.exists(src_dir):
    sys.path.insert(0, src_dir)

load_dotenv()

# ==================== PREDICTION FRESHNESS FUNCTIONS ====================
@st.cache_data(ttl=300)  # 5 minute cache
def check_prediction_freshness():
    """Check if predictions are fresh (<3 hours old)"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return "error", "No database connection", None
        
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        
        client = MongoClient(uri)
        db = client[db_name]
        
        # Check ensemble forecasts
        latest = db.ensemble_forecasts_3day.find_one(
            sort=[('created_at', -1)]
        )
        
        client.close()
        
        if not latest:
            return "no_data", "No predictions found", None
        
        created_at = latest.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', ''))
        
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
    """Trigger background prediction update"""
    try:
        # Try to import prediction service
        project_root = os.path.dirname(os.path.dirname(current_dir))
        model_training_dir = os.path.join(project_root, 'model_training')
        
        if os.path.exists(model_training_dir):
            sys.path.insert(0, model_training_dir)
            
            try:
                from prediction_service import PredictionService
                service = PredictionService()
                success = service.trigger_async_prediction_update()
                return success
            except ImportError:
                # Try to run the script directly
                import subprocess
                script_path = os.path.join(model_training_dir, 'prediction_service.py')
                if os.path.exists(script_path):
                    subprocess.Popen([sys.executable, script_path])
                    return True
        
        return False
    except:
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

# Helper function to ensure datetime
def ensure_datetime(timestamp):
    """Convert timestamp to datetime object if it's a string"""
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

# Page configuration
st.set_page_config(
    page_title="AQI Karachi - Air Quality Prediction",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .warning-card {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
        margin-bottom: 1rem;
    }
    .danger-card {
        background-color: #FEE2E2;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #EF4444;
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
    .feature-importance-bar { border-left: 5px solid #EC4899; }
</style>
""", unsafe_allow_html=True)

# Helper functions
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

# ==================== DATA LOADING FUNCTIONS ====================
@st.cache_data(ttl=3600)
def load_current_aqi():
    """Load current AQI data from MongoDB"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            st.error("MONGODB_URI not set in .env file")
            return None
            
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        
        client = MongoClient(uri)
        db = client[db_name]
        
        collections = db.list_collection_names()
        
        if 'aqi_measurements' in collections:
            latest = db.aqi_measurements.find_one(sort=[('timestamp', -1)])
            collection_name = 'aqi_measurements'
        elif 'aqi_features' in collections:
            latest = db.aqi_features.find_one(sort=[('timestamp', -1)])
            collection_name = 'aqi_features'
        else:
            st.warning(f"No AQI collections found. Available collections: {collections}")
            client.close()
            return None
        
        if latest:
            aqi_field = 'aqi' if 'aqi' in latest else 'us_aqi'
            pm25_field = 'pm25' if 'pm25' in latest else 'pm2_5'
            
            current_aqi = latest.get(aqi_field, 0)
            pm25 = latest.get(pm25_field, 0)
            pm10 = latest.get('pm10', 0)
            timestamp = latest.get('timestamp', datetime.now())
            
            timestamp = ensure_datetime(timestamp)
            
            return {
                'aqi': current_aqi,
                'pm25': pm25,
                'pm10': pm10,
                'timestamp': timestamp,
                'category': get_aqi_category(current_aqi)[0],
                'source': collection_name
            }
        
        client.close()
        return None
        
    except Exception as e:
        st.error(f"Error loading current AQI: {str(e)[:100]}")
        return None

@st.cache_data(ttl=3600)
def load_historical_data(days=60):
    """Load historical data for EDA"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            st.error("MONGODB_URI not set in .env file")
            return pd.DataFrame()
            
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        
        client = MongoClient(uri)
        db = client[db_name]
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        collections = db.list_collection_names()
        
        if 'aqi_measurements' in collections:
            historical_data = list(db.aqi_measurements.find({
                'timestamp': {'$gte': cutoff_date.isoformat()}
            }).sort('timestamp', 1))
            source_collection = 'aqi_measurements'
        elif 'aqi_features' in collections:
            historical_data = list(db.aqi_features.find({
                'timestamp': {'$gte': cutoff_date}
            }).sort('timestamp', 1))
            source_collection = 'aqi_features'
        else:
            st.warning(f"No data collections found. Available: {collections}")
            client.close()
            return pd.DataFrame()
        
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
                df['week'] = df['timestamp'].dt.isocalendar().week
            
            client.close()
            return df
        
        client.close()
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)[:100]}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_ml_forecast():
    """Load ML model forecasts"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return pd.DataFrame()
            
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        
        client = MongoClient(uri)
        db = client[db_name]
        
        forecast_collections = [
            'ml_forecasts_3day',
            'ml_forecasts', 
            'aqi_predictions', 
            'forecasts'
        ]
        
        for coll_name in forecast_collections:
            if coll_name in db.list_collection_names():
                forecast_data = list(db[coll_name].find({}, sort=[('date', 1)]))
                if forecast_data:
                    df = pd.DataFrame(forecast_data)
                    
                    if 'date' in df.columns:
                        try:
                            df['timestamp'] = pd.to_datetime(df['date'])
                        except:
                            df['timestamp'] = df['date']
                    elif 'timestamp' in df.columns:
                        df['date'] = df['timestamp'].dt.date
                    
                    df = df.sort_values('timestamp')
                    df['source'] = 'ML Model'
                    
                    if 'predicted_aqi' in df.columns:
                        df['predicted_aqi'] = pd.to_numeric(df['predicted_aqi'], errors='coerce')
                        df['predicted_aqi'] = df['predicted_aqi'].round(1)
                    
                    client.close()
                    return df
        
        client.close()
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error loading ML forecast: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_time_series_forecast():
    """Load time series forecasts"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return pd.DataFrame()
            
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        
        client = MongoClient(uri)
        db = client[db_name]
        
        forecast_collections = [
            'timeseries_forecasts_3day',
            'simple_forecasts', 
            'time_series_forecasts', 
            'ts_forecasts'
        ]
        
        for coll_name in forecast_collections:
            if coll_name in db.list_collection_names():
                forecast_data = list(db[coll_name].find({}, sort=[('date', 1)]))
                if forecast_data:
                    df = pd.DataFrame(forecast_data)
                    
                    if 'date' in df.columns:
                        try:
                            df['timestamp'] = pd.to_datetime(df['date'])
                        except:
                            df['timestamp'] = df['date']
                    elif 'timestamp' in df.columns:
                        df['date'] = df['timestamp'].dt.date
                    
                    df = df.sort_values('timestamp')
                    df['source'] = 'Time Series'
                    
                    if 'predicted_aqi' in df.columns:
                        df['predicted_aqi'] = pd.to_numeric(df['predicted_aqi'], errors='coerce')
                        df['predicted_aqi'] = df['predicted_aqi'].round(1)
                    
                    client.close()
                    return df
        
        client.close()
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error loading time series forecast: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_ensemble_forecast():
    """Load ensemble forecasts separately"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return pd.DataFrame()
            
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        
        client = MongoClient(uri)
        db = client[db_name]
        
        if 'ensemble_forecasts_3day' in db.list_collection_names():
            forecast_data = list(db.ensemble_forecasts_3day.find({}, sort=[('date', 1)]))
            if forecast_data:
                df = pd.DataFrame(forecast_data)
                
                if 'date' in df.columns:
                    try:
                        df['timestamp'] = pd.to_datetime(df['date'])
                    except:
                        df['timestamp'] = df['date']
                    df = df.sort_values('timestamp')
                    df['source'] = 'Ensemble (Recommended)'
                
                client.close()
                return df
        
        client.close()
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error loading ensemble forecast: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_model_metrics():
    """Load model performance metrics"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            return pd.DataFrame()
            
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        
        client = MongoClient(uri)
        db = client[db_name]
        
        metrics_data = []
        
        if 'model_registry' in db.list_collection_names():
            model_records = db.model_registry.find({})
            
            for model in model_records:
                metrics = model.get('metrics', {})
                if isinstance(metrics, dict):
                    created_at = model.get('created_at', datetime.now())
                    created_at = ensure_datetime(created_at)
                    
                    metrics_data.append({
                        'model_name': model.get('model_name', 'Unknown'),
                        'model_type': model.get('model_type', 'ML Model'),
                        'r2_score': metrics.get('Test R²', 0),
                        'mae': metrics.get('Test MAE', 0),
                        'rmse': metrics.get('Test RMSE', 0),
                        'created_at': created_at,
                        'is_production': model.get('is_production', False),
                        'model_id': model.get('model_id', '')
                    })
        
        client.close()
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            return df.sort_values('r2_score', ascending=False)
        else:
            return pd.DataFrame()
        
    except Exception as e:
        return pd.DataFrame()

# ==================== FEATURE IMPORTANCE FUNCTIONS ====================
@st.cache_data(ttl=3600)
def load_feature_importance():
    """Load feature importance from ML model"""
    try:
        import joblib
        
        # Look for the latest ML model
        models_dir = 'models'
        if not os.path.exists(models_dir):
            return None
        
        # Find the latest ML model file
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
        if not model_files:
            return None
        
        # Load the latest model
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
        model_path = os.path.join(models_dir, latest_model)
        
        # Load model data
        model_data = joblib.load(model_path)
        
        if 'model' in model_data and 'feature_names' in model_data:
            return model_data
        else:
            return None
            
    except Exception as e:
        print(f"Error loading feature importance: {e}")
        return None

def calculate_feature_importance(model_data):
    """Calculate feature importance for different model types"""
    try:
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        # Get feature importances based on model type
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances,
                'importance_abs': np.abs(importances)
            }).sort_values('importance_abs', ascending=False)
            
            return {
                'type': 'tree_importance',
                'data': importance_df,
                'method': 'Feature Importances (Gini Importance)'
            }
        
        elif hasattr(model, 'coef_'):
            # For linear models
            coefficients = model.coef_.flatten()
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients,
                'importance': np.abs(coefficients)
            }).sort_values('importance', ascending=False)
            
            return {
                'type': 'coefficients',
                'data': importance_df,
                'method': 'Coefficient Magnitudes'
            }
        
        else:
            # For other models
            return {
                'type': 'generic',
                'data': None,
                'method': 'Feature importance not available'
            }
            
    except Exception as e:
        return {
            'type': 'error',
            'data': None,
            'method': f"Error: {str(e)[:100]}"
        }

def create_feature_importance_plot(importance_data):
    """Create feature importance visualization"""
    if importance_data['type'] in ['tree_importance', 'coefficients'] and importance_data['data'] is not None:
        df = importance_data['data'].head(15)  # Show top 15 features
        
        fig = go.Figure()
        
        if importance_data['type'] == 'coefficients':
            # Color by positive/negative coefficients
            colors = ['#10B981' if x > 0 else '#EF4444' for x in df['coefficient']]
            fig.add_trace(go.Bar(
                x=df['importance'],
                y=df['feature'],
                orientation='h',
                marker_color=colors,
                text=[f"{coef:.4f}" for coef in df['coefficient']],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Coefficient: %{text}<br>Magnitude: %{x:.4f}<extra></extra>'
            ))
            title = 'Top 15 Feature Coefficients (Absolute Value)'
            xaxis_title = 'Coefficient (Absolute Value)'
        else:
            # Tree-based importance
            fig.add_trace(go.Bar(
                x=df['importance'],
                y=df['feature'],
                orientation='h',
                marker_color='#3B82F6',
                text=[f"{imp:.4f}" for imp in df['importance']],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            ))
            title = 'Top 15 Feature Importances'
            xaxis_title = 'Importance Score'
        
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title='Feature',
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    else:
        return None

# ==================== SIDEBAR NAVIGATION ====================
st.sidebar.title("🌫️ AQI Karachi Dashboard")

# Check prediction freshness for sidebar display
freshness_status, freshness_msg, last_updated = check_prediction_freshness()
freshness_icon = get_freshness_icon(freshness_status)

st.sidebar.markdown(f"""
### Prediction Status
<div class="freshness-{freshness_status}" style="padding: 10px; border-radius: 5px; margin-bottom: 20px;">
    {freshness_icon} **{freshness_msg}**
</div>
""", unsafe_allow_html=True)

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

# Update button in sidebar
if st.sidebar.button("🔄 Update Predictions Now", use_container_width=True, type="secondary"):
    with st.sidebar:
        with st.spinner("Updating predictions..."):
            if trigger_prediction_update():
                st.success("Update triggered!")
                time.sleep(2)
                st.rerun()
            else:
                st.error("Failed to trigger update")

# Refresh all data button
if st.sidebar.button("🗂️ Refresh All Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("**Karachi AQI Prediction System**\n\nReal-time air quality forecasting using 45-day Open-Meteo data.")

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🌫️ AQI Karachi - Air Quality Prediction System</h1>', unsafe_allow_html=True)
    
    # System status banner
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
            if trigger_prediction_update():
                st.success("Update triggered!")
                time.sleep(2)
                st.rerun()
    
    with col3:
        if st.button("🔍 Check Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
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
    3. **Feature Importance** - Understand what drives AQI predictions
    4. **Historical Trends** - Data visualization and analysis
    5. **3-Day Forecast** - Predictive analytics for future AQI (always fresh!)
    6. **Health Recommendations** - Precautions based on AQI levels
    7. **Model Comparison** - Performance metrics of different algorithms
    
    ### 🚀 How It Works:
    1. **Data Collection**: Every 3 hours → MongoDB
    2. **Feature Engineering**: Extract time-based features
    3. **Model Training**: Twice daily → Time series and ML models
    4. **Prediction Generation**: Every 3 hours → Fresh 3-day forecasts
    5. **Dashboard**: Visualize results with freshness check
    """)
    
    # Quick stats
    st.markdown("### 📊 Quick Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_data = load_current_aqi()
        if current_data:
            st.metric("Current AQI", f"{current_data['aqi']:.0f}", 
                     delta=f"{current_data['category']}")
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
            best_r2 = metrics_data['r2_score'].max()
            st.metric("Best R²", f"{best_r2:.3f}")
        else:
            st.metric("Best R²", "N/A")
    
    # Freshness explanation
    with st.expander("ℹ️ About Prediction Freshness"):
        st.markdown("""
        **What does freshness mean?**
        
        - **✅ Fresh (<3 hours)**: Predictions are up-to-date
        - **⚠️ Stale (3-6 hours)**: Predictions are getting old
        - **🔄 Very Stale (6-12 hours)**: Predictions need updating
        - **❌ Outdated (>12 hours)**: Predictions are too old
        
        **Automatic Updates:**
        - Data collection: Every 3 hours
        - Model training: Twice daily (3 AM & 3 PM)
        - Predictions: Every 3 hours
        
        **Manual Updates:**
        Click the "Update Now" button to generate fresh predictions immediately!
        """)

# ==================== CURRENT AQI PAGE ====================
elif page == "📊 Current AQI":
    st.markdown('<h1 class="main-header">📊 Current Air Quality in Karachi</h1>', unsafe_allow_html=True)
    
    current_data = load_current_aqi()
    
    if current_data:
        # AQI Display
        aqi_value = current_data['aqi']
        category, color, emoji = get_aqi_category(aqi_value)
        
        timestamp = current_data['timestamp']
        if isinstance(timestamp, datetime):
            display_time = timestamp.strftime('%Y-%m-%d %H:%M')
        else:
            display_time = str(timestamp)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Current AQI</h3>
                <h1 style="color: {color}; font-size: 3rem;">{aqi_value:.0f}</h1>
                <h3>{emoji} {category}</h3>
                <p>Last updated: {display_time}</p>
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
        
        # Health Precautions
        st.markdown(f'<h2 class="sub-header">🩺 Health Precautions for {category} Air Quality</h2>', unsafe_allow_html=True)
        
        precautions = get_precautions(category)
        for precaution in precautions:
            st.markdown(f"• {precaution}")
        
        # Data source info
        if 'source' in current_data:
            st.info(f"Data source: {current_data['source']} collection")
            
    else:
        st.warning("No current AQI data available. Please run data collection first.")

# ==================== EDA ANALYSIS PAGE (FULL VERSION) ====================
elif page == "📈 EDA Analysis":
    st.markdown('<h1 class="main-header">📈 Exploratory Data Analysis (EDA)</h1>', unsafe_allow_html=True)
    
    # Tab layout for different EDA sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview", 
        "📈 AQI Analysis", 
        "⏰ Temporal Patterns", 
        "🔗 Correlations", 
        "🔍 Outliers & Anomalies", 
        "📋 Summary Report"
    ])
    
    # Load data with adjustable days
    with st.sidebar:
        st.markdown("### ⚙️ EDA Settings")
        days_to_load = st.slider("Days to analyze:", 7, 180, 45, key="eda_days")
    
    hist_data = load_historical_data(days_to_load)
    
    if not hist_data.empty:
        # ========== TAB 1: DATA OVERVIEW ==========
        with tab1:
            st.markdown('<h2 class="sub-header">📊 Data Overview</h2>', unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{len(hist_data):,}")
            
            with col2:
                if 'timestamp' in hist_data.columns:
                    time_diff = hist_data['timestamp'].max() - hist_data['timestamp'].min()
                    st.metric("Time Period", f"{time_diff.days} days")
                else:
                    st.metric("Time Period", "N/A")
            
            with col3:
                if 'aqi' in hist_data.columns:
                    avg_aqi = hist_data['aqi'].mean()
                    st.metric("Avg AQI", f"{avg_aqi:.1f}")
                else:
                    st.metric("Avg AQI", "N/A")
            
            with col4:
                completeness = (1 - hist_data.isnull().sum().sum() / (len(hist_data) * len(hist_data.columns))) * 100
                st.metric("Data Completeness", f"{completeness:.1f}%")
            
            # Column Information
            st.markdown("### 📋 Column Information")
            col_info = pd.DataFrame({
                'Column': hist_data.columns,
                'Data Type': hist_data.dtypes.astype(str),
                'Non-Null': hist_data.count().values,
                'Null %': (hist_data.isnull().sum() / len(hist_data) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
            
            # Data Sample
            with st.expander("👀 View Data Sample"):
                st.dataframe(hist_data.head(20), use_container_width=True)
            
            # Statistical Summary
            st.markdown("### 📈 Statistical Summary")
            numeric_cols = hist_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary_stats = hist_data[numeric_cols].describe().T
                
                # Calculate skewness and kurtosis manually
                def simple_skew(series):
                    mean = series.mean()
                    std = series.std()
                    if std == 0:
                        return 0
                    n = len(series)
                    return ((series - mean)**3).sum() / (n * std**3)
                
                def simple_kurt(series):
                    mean = series.mean()
                    std = series.std()
                    if std == 0:
                        return 0
                    n = len(series)
                    return ((series - mean)**4).sum() / (n * std**4) - 3
                
                try:
                    summary_stats['skew'] = hist_data[numeric_cols].apply(simple_skew)
                    summary_stats['kurtosis'] = hist_data[numeric_cols].apply(simple_kurt)
                except:
                    summary_stats['skew'] = 0
                    summary_stats['kurtosis'] = 0
                
                st.dataframe(summary_stats.round(4), use_container_width=True)
            else:
                st.warning("No numeric columns found for statistical summary")
        
        # ========== TAB 2: AQI ANALYSIS ==========
        with tab2:
            if 'aqi' in hist_data.columns:
                aqi_data = hist_data['aqi'].dropna()
                
                st.markdown('<h2 class="sub-header">🎯 AQI Distribution Analysis</h2>', unsafe_allow_html=True)
                
                # Create two columns for AQI charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram with KDE
                    fig1 = px.histogram(hist_data, x='aqi', nbins=50, 
                                       marginal="box",
                                       title='AQI Distribution with Box Plot',
                                       labels={'aqi': 'AQI Value'},
                                       color_discrete_sequence=['#3B82F6'],
                                       opacity=0.7)
                    fig1.update_layout(height=400)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Violin plot
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
                
                # AQI Categories Distribution
                st.markdown("### 🏷️ AQI Categories Distribution")
                
                # Categorize AQI
                def categorize_aqi_value(x):
                    try:
                        x = float(x)
                        if x <= 50:
                            return "Good"
                        elif x <= 100:
                            return "Moderate"
                        elif x <= 150:
                            return "Unhealthy for Sensitive"
                        elif x <= 200:
                            return "Unhealthy"
                        elif x <= 300:
                            return "Very Unhealthy"
                        else:
                            return "Hazardous"
                    except:
                        return "Unknown"
                
                hist_data['aqi_category'] = hist_data['aqi'].apply(categorize_aqi_value)
                category_counts = hist_data['aqi_category'].value_counts().reset_index()
                category_counts.columns = ['Category', 'Count']
                category_counts['Percentage'] = (category_counts['Count'] / len(hist_data) * 100).round(2)
                
                # Colors for categories
                category_colors = {
                    "Good": "#10B981",
                    "Moderate": "#F59E0B",
                    "Unhealthy for Sensitive": "#F97316",
                    "Unhealthy": "#EF4444",
                    "Very Unhealthy": "#8B5CF6",
                    "Hazardous": "#7C3AED",
                    "Unknown": "#6B7280"
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig3 = px.pie(category_counts, values='Count', names='Category',
                                 title='AQI Category Distribution',
                                 color='Category',
                                 color_discrete_map=category_colors)
                    fig3.update_layout(height=400)
                    st.plotly_chart(fig3, use_container_width=True)
                
                with col2:
                    # Bar chart
                    fig4 = px.bar(category_counts, x='Category', y='Percentage',
                                 title='AQI Category Percentage',
                                 color='Category',
                                 color_discrete_map=category_colors,
                                 text='Percentage')
                    fig4.update_layout(height=400, yaxis_title='Percentage (%)')
                    st.plotly_chart(fig4, use_container_width=True)
                
                # Cumulative Distribution
                st.markdown("### 📊 Cumulative Distribution")
                sorted_aqi = np.sort(aqi_data)
                cdf = np.arange(1, len(sorted_aqi) + 1) / len(sorted_aqi)
                
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(x=sorted_aqi, y=cdf,
                                         mode='lines',
                                         name='CDF',
                                         line=dict(color='green', width=2)))
                fig5.update_layout(title='AQI Cumulative Distribution Function',
                                  xaxis_title='AQI',
                                  yaxis_title='Cumulative Probability',
                                  height=400)
                st.plotly_chart(fig5, use_container_width=True)
                
            else:
                st.warning("AQI data not available for analysis")
        
        # ========== TAB 3: TEMPORAL PATTERNS ==========
        with tab3:
            if 'timestamp' in hist_data.columns and 'aqi' in hist_data.columns:
                st.markdown('<h2 class="sub-header">⏰ Temporal Pattern Analysis</h2>', unsafe_allow_html=True)
                
                # Daily Trend
                st.markdown("### 📅 Daily AQI Trend")
                hist_data['date'] = hist_data['timestamp'].dt.date
                daily_avg = hist_data.groupby('date')['aqi'].agg(['mean', 'std', 'min', 'max']).reset_index()
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=daily_avg['date'], y=daily_avg['mean'],
                                         mode='lines',
                                         name='Daily Avg',
                                         line=dict(color='blue', width=2)))
                fig1.add_trace(go.Scatter(x=daily_avg['date'], 
                                         y=daily_avg['mean'] + daily_avg['std'],
                                         mode='lines',
                                         name='+1 Std Dev',
                                         line=dict(color='lightblue', width=1, dash='dash'),
                                         showlegend=False))
                fig1.add_trace(go.Scatter(x=daily_avg['date'],
                                         y=daily_avg['mean'] - daily_avg['std'],
                                         mode='lines',
                                         name='-1 Std Dev',
                                         fill='tonexty',
                                         fillcolor='rgba(173, 216, 230, 0.3)',
                                         line=dict(color='lightblue', width=1, dash='dash'),
                                         showlegend=False))
                fig1.update_layout(title='Daily AQI Trend with Standard Deviation',
                                  xaxis_title='Date',
                                  yaxis_title='AQI',
                                  height=400)
                st.plotly_chart(fig1, use_container_width=True)
                
                # Hourly Pattern
                st.markdown("### 🕒 Hourly Pattern")
                hist_data['hour'] = hist_data['timestamp'].dt.hour
                hourly_avg = hist_data.groupby('hour')['aqi'].agg(['mean', 'std']).reset_index()
                
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=hourly_avg['hour'], y=hourly_avg['mean'],
                                     name='Avg AQI',
                                     marker_color='orange',
                                     error_y=dict(type='data', array=hourly_avg['std'], visible=True)))
                fig2.update_layout(title='Average AQI by Hour of Day',
                                  xaxis_title='Hour',
                                  yaxis_title='Average AQI',
                                  height=400)
                st.plotly_chart(fig2, use_container_width=True)
                
                # Weekly Pattern
                st.markdown("### 📆 Weekly Pattern")
                hist_data['day_of_week'] = hist_data['timestamp'].dt.dayofweek
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                hist_data['day_name'] = hist_data['day_of_week'].apply(lambda x: day_names[x])
                
                weekly_avg = hist_data.groupby('day_name')['aqi'].agg(['mean', 'std']).reset_index()
                weekly_avg['day_name'] = pd.Categorical(weekly_avg['day_name'], 
                                                       categories=day_names, 
                                                       ordered=True)
                weekly_avg = weekly_avg.sort_values('day_name')
                
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(x=weekly_avg['day_name'], y=weekly_avg['mean'],
                                     name='Avg AQI',
                                     marker_color='green',
                                     error_y=dict(type='data', array=weekly_avg['std'], visible=True)))
                fig3.update_layout(title='Average AQI by Day of Week',
                                  xaxis_title='Day of Week',
                                  yaxis_title='Average AQI',
                                  height=400)
                st.plotly_chart(fig3, use_container_width=True)
                
                # Monthly Pattern
                st.markdown("### 📅 Monthly Pattern")
                hist_data['month'] = hist_data['timestamp'].dt.month
                monthly_avg = hist_data.groupby('month')['aqi'].agg(['mean', 'std']).reset_index()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                fig4 = go.Figure()
                fig4.add_trace(go.Bar(x=[month_names[m-1] for m in monthly_avg['month']],
                                     y=monthly_avg['mean'],
                                     name='Avg AQI',
                                     marker_color='purple',
                                     error_y=dict(type='data', array=monthly_avg['std'], visible=True)))
                fig4.update_layout(title='Average AQI by Month',
                                  xaxis_title='Month',
                                  yaxis_title='Average AQI',
                                  height=400)
                st.plotly_chart(fig4, use_container_width=True)
                
            else:
                st.warning("Timestamp or AQI data not available for temporal analysis")
        
        # ========== TAB 4: CORRELATIONS ==========
        with tab4:
            st.markdown('<h2 class="sub-header">🔗 Correlation Analysis</h2>', unsafe_allow_html=True)
            
            # Select numeric columns
            numeric_cols = hist_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                # Correlation matrix
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
                
                # Top correlations with AQI
                if 'aqi' in numeric_cols:
                    st.markdown("### 🔝 Top Correlations with AQI")
                    aqi_corr = corr_matrix['aqi'].drop('aqi', errors='ignore').sort_values(ascending=False)
                    
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=aqi_corr.head(10).values,
                        y=aqi_corr.head(10).index,
                        orientation='h',
                        marker_color=['green' if x > 0 else 'red' for x in aqi_corr.head(10).values]
                    ))
                    fig2.update_layout(title='Top 10 Features Correlated with AQI',
                                      xaxis_title='Correlation Coefficient',
                                      yaxis_title='Feature',
                                      height=400)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Correlation statistics
                    with st.expander("📈 View Correlation Statistics"):
                        corr_stats = pd.DataFrame({
                            'Feature': aqi_corr.index,
                            'Correlation': aqi_corr.values,
                            'Strength': ['Strong' if abs(x) > 0.7 else 'Moderate' if abs(x) > 0.3 else 'Weak' 
                                        for x in aqi_corr.values]
                        })
                        st.dataframe(corr_stats, use_container_width=True)
                
            else:
                st.warning("Not enough numeric columns for correlation analysis")
        
        # ========== TAB 5: OUTLIERS & ANOMALIES ==========
        with tab5:
            if 'aqi' in hist_data.columns:
                aqi_data = hist_data['aqi'].dropna()
                
                st.markdown('<h2 class="sub-header">🔍 Outlier & Anomaly Detection</h2>', unsafe_allow_html=True)
                
                # Calculate outliers using IQR method
                Q1 = aqi_data.quantile(0.25)
                Q3 = aqi_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = hist_data[(hist_data['aqi'] < lower_bound) | (hist_data['aqi'] > upper_bound)]
                
                # Outlier statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Outliers", len(outliers))
                
                with col2:
                    outlier_percent = (len(outliers) / len(hist_data)) * 100
                    st.metric("Outlier %", f"{outlier_percent:.2f}%")
                
                with col3:
                    st.metric("Lower Bound", f"{lower_bound:.1f}")
                
                with col4:
                    st.metric("Upper Bound", f"{upper_bound:.1f}")
                
                # Box plot showing outliers
                fig1 = go.Figure()
                fig1.add_trace(go.Box(y=aqi_data,
                                     name='AQI',
                                     boxpoints='outliers',
                                     marker_color='blue'))
                fig1.update_layout(title='AQI Box Plot with Outliers',
                                  yaxis_title='AQI',
                                  height=400)
                st.plotly_chart(fig1, use_container_width=True)
                
                # Time series with outliers highlighted
                if 'timestamp' in hist_data.columns:
                    fig2 = go.Figure()
                    
                    # Normal points
                    normal_points = hist_data[~hist_data.index.isin(outliers.index)]
                    fig2.add_trace(go.Scatter(
                        x=normal_points['timestamp'],
                        y=normal_points['aqi'],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='blue', size=3, opacity=0.5)
                    ))
                    
                    # Outlier points
                    if len(outliers) > 0:
                        fig2.add_trace(go.Scatter(
                            x=outliers['timestamp'],
                            y=outliers['aqi'],
                            mode='markers',
                            name='Outliers',
                            marker=dict(color='red', size=6, symbol='x')
                        ))
                    
                    fig2.update_layout(title='AQI Time Series with Outliers Highlighted',
                                      xaxis_title='Time',
                                      yaxis_title='AQI',
                                      height=400)
                    st.plotly_chart(fig2, use_container_width=True)
                
                # View outliers table
                with st.expander("📋 View Detected Outliers"):
                    if len(outliers) > 0:
                        outliers_display = outliers.copy()
                        if 'timestamp' in outliers_display.columns:
                            outliers_display['timestamp'] = outliers_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                        st.dataframe(outliers_display[['timestamp', 'aqi']].head(20), use_container_width=True)
                    else:
                        st.info("No outliers detected")
                
            else:
                st.warning("AQI data not available for outlier analysis")
        
        # ========== TAB 6: SUMMARY REPORT ==========
        with tab6:
            st.markdown('<h1 class="sub-header">📋 EDA Summary Report</h1>', unsafe_allow_html=True)
            
            if 'aqi' in hist_data.columns:
                aqi_data = hist_data['aqi'].dropna()
                
                # Key Statistics
                st.markdown("### 📊 Key Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean AQI", f"{aqi_data.mean():.1f}")
                    st.metric("Median AQI", f"{aqi_data.median():.1f}")
                
                with col2:
                    st.metric("Std Deviation", f"{aqi_data.std():.1f}")
                    st.metric("Variance", f"{aqi_data.var():.1f}")
                
                with col3:
                    # Calculate skewness manually
                    n = len(aqi_data)
                    if n > 0 and aqi_data.std() > 0:
                        skewness = ((aqi_data - aqi_data.mean())**3).sum() / (n * aqi_data.std()**3)
                        st.metric("Skewness", f"{skewness:.2f}")
                    else:
                        st.metric("Skewness", "N/A")
                    
                    # Calculate kurtosis manually
                    if n > 0 and aqi_data.std() > 0:
                        kurtosis = ((aqi_data - aqi_data.mean())**4).sum() / (n * aqi_data.std()**4) - 3
                        st.metric("Kurtosis", f"{kurtosis:.2f}")
                    else:
                        st.metric("Kurtosis", "N/A")
                
                with col4:
                    st.metric("Range", f"{aqi_data.max() - aqi_data.min():.1f}")
                    st.metric("IQR", f"{aqi_data.quantile(0.75) - aqi_data.quantile(0.25):.1f}")
                
                # Insights
                st.markdown("### 💡 Key Insights")
                
                insights = []
                
                # Data quality insights
                missing_cols = hist_data.isnull().sum()
                high_missing = missing_cols[missing_cols > 0]
                if len(high_missing) > 0:
                    insights.append(f"⚠️ **Data Quality**: {len(high_missing)} columns have missing values")
                
                # AQI distribution insights
                if aqi_data.std() > 0:
                    skew_val = ((aqi_data - aqi_data.mean())**3).sum() / (len(aqi_data) * aqi_data.std()**3)
                    if abs(skew_val) > 1:
                        skew_direction = "right" if skew_val > 0 else "left"
                        insights.append(f"📊 **Distribution**: AQI is highly skewed to the {skew_direction} (skewness = {skew_val:.2f})")
                    elif abs(skew_val) > 0.5:
                        skew_direction = "right" if skew_val > 0 else "left"
                        insights.append(f"📊 **Distribution**: AQI is moderately skewed to the {skew_direction}")
                    else:
                        insights.append("✅ **Distribution**: AQI is approximately symmetric")
                else:
                    insights.append("📊 **Distribution**: Not enough data for distribution analysis")
                
                # Outlier insights
                outlier_percent = (len(outliers) / len(hist_data)) * 100 if 'outliers' in locals() else 0
                if outlier_percent > 10:
                    insights.append(f"🔍 **Outliers**: High percentage of outliers ({outlier_percent:.1f}%) detected")
                elif outlier_percent > 5:
                    insights.append(f"🔍 **Outliers**: Moderate percentage of outliers ({outlier_percent:.1f}%) detected")
                else:
                    insights.append("✅ **Outliers**: Minimal outliers detected")
                
                # Temporal pattern insights
                if 'hour' in hist_data.columns:
                    hourly_avg = hist_data.groupby('hour')['aqi'].mean()
                    hour_range = hourly_avg.max() - hourly_avg.min()
                    if hour_range > 30:
                        insights.append(f"⏰ **Daily Pattern**: Strong daily variation (range = {hour_range:.1f} AQI)")
                    elif hour_range > 15:
                        insights.append(f"⏰ **Daily Pattern**: Moderate daily variation (range = {hour_range:.1f} AQI)")
                
                # Correlation insights
                if 'corr_matrix' in locals() and 'aqi' in corr_matrix.columns:
                    aqi_corr = corr_matrix['aqi'].drop('aqi', errors='ignore')
                    strong_corr = aqi_corr[abs(aqi_corr) > 0.7]
                    if len(strong_corr) > 0:
                        insights.append(f"🔗 **Correlations**: {len(strong_corr)} features strongly correlated with AQI")
                
                # Display insights
                for insight in insights:
                    st.markdown(f"- {insight}")
                
                # Recommendations
                st.markdown("### 🚀 Recommendations")
                
                recommendations = [
                    "✅ **Data Collection**: Continue collecting data for better model performance",
                    "✅ **Feature Engineering**: Create time-based features (hour of day, day of week)",
                    "✅ **Model Selection**: Consider time series models for forecasting",
                    "✅ **Monitoring**: Set up alerts for extreme AQI values",
                    "✅ **Validation**: Use cross-validation for model evaluation"
                ]
                
                if outlier_percent > 10:
                    recommendations.append("⚠️ **Outlier Handling**: Consider outlier removal or robust models")
                
                if len(high_missing) > 0:
                    recommendations.append("⚠️ **Missing Data**: Implement imputation strategies")
                
                for rec in recommendations:
                    st.markdown(f"- {rec}")
                
                # Export option
                st.markdown("### 💾 Export Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    summary_text = f"""
                    AQI Karachi - EDA Summary Report
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    Dataset Summary:
                    - Records: {len(hist_data):,}
                    - Time Period: {days_to_load} days
                    - Avg AQI: {aqi_data.mean():.1f}
                    - Std Dev: {aqi_data.std():.1f}
                    
                    Key Insights:
                    {chr(10).join(insights)}
                    
                    Recommendations:
                    {chr(10).join([r.replace('✅ ', '').replace('⚠️ ', '') for r in recommendations])}
                    """
                    
                    st.download_button(
                        label="⬇️ Download Report",
                        data=summary_text,
                        file_name=f"aqi_eda_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    if st.button("📊 Export Charts", use_container_width=True):
                        st.info("Chart export feature coming soon!")
                        
            else:
                st.warning("AQI data not available for summary report")
    
    else:
        st.warning("No historical data available for EDA.")

# ==================== FEATURE IMPORTANCE PAGE ====================
elif page == "🎯 Feature Importance":
    st.markdown('<h1 class="main-header">🎯 Feature Importance Analysis</h1>', unsafe_allow_html=True)
    
    # Load ML model and feature importance
    model_data = load_feature_importance()
    
    if model_data:
        st.success(f"✅ Loaded model from: {os.path.join('models', max(os.listdir('models'), key=lambda x: os.path.getctime(os.path.join('models', x))))}")
        
        # Display model information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_type = type(model_data['model']).__name__
            st.metric("Model Type", model_type)
        
        with col2:
            if 'train_r2' in model_data:
                st.metric("Train R²", f"{model_data['train_r2']:.4f}")
            else:
                st.metric("Train R²", "N/A")
        
        with col3:
            if 'test_r2' in model_data:
                st.metric("Test R²", f"{model_data['test_r2']:.4f}")
            else:
                st.metric("Test R²", "N/A")
        
        # Calculate feature importance
        importance_data = calculate_feature_importance(model_data)
        
        if importance_data['data'] is not None:
            # Feature importance plot
            st.markdown('<h2 class="sub-header">📊 Feature Importance Visualization</h2>', unsafe_allow_html=True)
            st.info(f"Method: {importance_data['method']}")
            
            fig = create_feature_importance_plot(importance_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.markdown("### 📋 Feature Importance Table")
            st.dataframe(importance_data['data'].head(20), use_container_width=True)
            
            # Feature descriptions
            st.markdown("### 📝 Feature Descriptions")
            feature_descriptions = {
                'hour': 'Hour of day (0-23)',
                'day_of_week': 'Day of week (0=Monday, 6=Sunday)',
                'hour_sin': 'Sine of hour (for cyclical encoding)',
                'hour_cos': 'Cosine of hour (for cyclical encoding)',
                'is_weekend': 'Weekend indicator (1=Weekend, 0=Weekday)',
                'is_morning': 'Morning hours (6 AM - 11 AM)',
                'is_afternoon': 'Afternoon hours (12 PM - 5 PM)',
                'is_evening': 'Evening hours (6 PM - 11 PM)',
                'lag_1h': 'AQI value 1 hour ago',
                'lag_2h': 'AQI value 2 hours ago',
                'lag_3h': 'AQI value 3 hours ago',
                'lag_6h': 'AQI value 6 hours ago',
                'lag_12h': 'AQI value 12 hours ago',
                'lag_24h': 'AQI value 24 hours ago',
                'lag_48h': 'AQI value 48 hours ago',
                'lag_72h': 'AQI value 72 hours ago',
                'rolling_mean_6h': '6-hour rolling average of AQI',
                'rolling_mean_24h': '24-hour rolling average of AQI',
                'rolling_mean_72h': '72-hour rolling average of AQI',
                'target_3day_avg': 'Target variable: 3-day average AQI'
            }
            
            # Display feature descriptions for top features
            top_features = importance_data['data'].head(10)['feature'].tolist()
            for feature in top_features:
                if feature in feature_descriptions:
                    st.markdown(f"**{feature}**: {feature_descriptions[feature]}")
                else:
                    st.markdown(f"**{feature}**: No description available")
            
            # Model insights
            st.markdown("### 💡 Model Insights")
            
            insights = []
            
            # Check for lag features importance
            lag_features = [f for f in top_features if f.startswith('lag_')]
            if lag_features:
                insights.append(f"⏰ **Time Dependency**: {len(lag_features)} lag features are important, indicating strong temporal patterns")
            
            # Check for time features
            time_features = ['hour', 'day_of_week', 'hour_sin', 'hour_cos', 'is_weekend']
            important_time_features = [f for f in top_features if f in time_features]
            if important_time_features:
                insights.append(f"🕒 **Time Patterns**: {len(important_time_features)} time-based features are important")
            
            # Check for rolling features
            rolling_features = [f for f in top_features if 'rolling' in f]
            if rolling_features:
                insights.append(f"📈 **Trend Importance**: {len(rolling_features)} rolling average features are important")
            
            # Check coefficient signs for linear models
            if importance_data['type'] == 'coefficients':
                positive_coefs = importance_data['data'][importance_data['data']['coefficient'] > 0].head(3)
                negative_coefs = importance_data['data'][importance_data['data']['coefficient'] < 0].head(3)
                
                if len(positive_coefs) > 0:
                    insights.append("📊 **Positive Influencers**: Features that increase predicted AQI")
                if len(negative_coefs) > 0:
                    insights.append("📉 **Negative Influencers**: Features that decrease predicted AQI")
            
            for insight in insights:
                st.markdown(f"- {insight}")
            
            # Download feature importance
            if st.button("📥 Download Feature Importance Data", use_container_width=True):
                csv = importance_data['data'].to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"feature_importance_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        else:
            st.warning("Feature importance data not available for this model type")
            st.info(f"Model type: {type(model_data['model']).__name__}")
    
    else:
        st.info("""
        ## 🤖 No Model Available for Feature Importance
        
        **To see feature importance:**
        
        1. **Train ML models first:**
        ```bash
        python model_training/combined_training.py
        ```
        
        or
        
        ```bash
        python model_training/train_models.py
        ```
        
        2. **Check if models were saved in 'models/' directory**
        
        3. **Refresh this page** after training
        
        **Note**: Feature importance is only available for models that support it (Random Forest, Linear Models, etc.)
        """)
# ==================== HISTORICAL TRENDS PAGE ====================
elif page == "📊 Historical Trends":
    st.markdown('<h1 class="main-header">📊 Historical AQI Trends</h1>', unsafe_allow_html=True)
    
    # Load data with adjustable days
    days_to_load = st.slider("Select number of days to analyze:", 7, 90, 30, key="history_days")
    hist_data = load_historical_data(days_to_load)
    
    if not hist_data.empty:
        # Time Series Plot - FIXED VERSION
        st.markdown('<h2 class="sub-header">⏰ AQI Time Series</h2>', unsafe_allow_html=True)
        
        if 'timestamp' in hist_data.columns and 'aqi' in hist_data.columns:
            # Use go.Figure instead of px.line for spline support
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=hist_data['timestamp'],
                y=hist_data['aqi'],
                mode='lines',
                name='AQI',
                line=dict(color='#3B82F6', width=2, shape='spline', smoothing=1.3),
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d %H:%M}<br><b>AQI:</b> %{y:.1f}<extra></extra>',
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)'
            ))
            
            fig.update_layout(
                title=f'AQI Trend - Last {days_to_load} Days',
                xaxis_title='Time',
                yaxis_title='AQI',
                height=400,
                hovermode='x unified',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Timestamp or AQI column not found in data")
        
        # Daily Averages
        st.markdown('<h2 class="sub-header">📅 Daily Averages</h2>', unsafe_allow_html=True)
        
        if 'timestamp' in hist_data.columns and 'aqi' in hist_data.columns:
            hist_data['date'] = hist_data['timestamp'].dt.date
            daily_avg = hist_data.groupby('date')['aqi'].agg(['mean', 'min', 'max']).reset_index()
            
            fig = go.Figure()
            
            # Mean line
            fig.add_trace(go.Scatter(
                x=daily_avg['date'],
                y=daily_avg['mean'],
                mode='lines+markers',
                name='Daily Average',
                line=dict(color='#3B82F6', width=3),
                marker=dict(size=6, color='#3B82F6'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Avg AQI:</b> %{y:.1f}<extra></extra>'
            ))
            
            # Min-Max range
            fig.add_trace(go.Scatter(
                x=daily_avg['date'],
                y=daily_avg['max'],
                mode='lines',
                name='Max',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=daily_avg['date'],
                y=daily_avg['min'],
                mode='lines',
                name='Min',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(59, 130, 246, 0.2)',
                showlegend=False,
                hovertemplate='<b>Date:</b> %{x}<br><b>Min AQI:</b> %{y:.1f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Daily AQI Statistics',
                xaxis_title='Date',
                yaxis_title='AQI',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # AQI Categories Over Time
        st.markdown('<h2 class="sub-header">🎨 AQI Categories Over Time</h2>', unsafe_allow_html=True)
        
        if 'aqi' in hist_data.columns:
            hist_data['aqi_category'] = hist_data['aqi'].apply(lambda x: get_aqi_category(x)[0])
            
            category_counts = hist_data['aqi_category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            fig = px.pie(category_counts, values='Count', names='Category',
                        title='Distribution of AQI Categories',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No historical data available.")

# ==================== 3-DAY FORECAST PAGE (UPDATED WITH FRESHNESS) ====================
elif page == "🔮 3-Day Forecast":
    st.markdown('<h1 class="main-header">🔮 3-Day AQI Forecast</h1>', unsafe_allow_html=True)
    
    # Check prediction freshness
    freshness_status, freshness_msg, last_updated = check_prediction_freshness()
    freshness_icon = get_freshness_icon(freshness_status)
    
    # Display freshness banner
    st.markdown(f"""
    <div class="freshness-{freshness_status}" style="padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <h4 style="margin: 0;">{freshness_icon} Forecast Status: {freshness_status.upper().replace('_', ' ')}</h4>
        <p style="margin: 5px 0 0 0;">{freshness_msg}</p>
        {f"<small>Last update: {last_updated.strftime('%Y-%m-%d %H:%M') if last_updated else 'Unknown'}</small>"}
    </div>
    """, unsafe_allow_html=True)
    
    # Control buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if freshness_status in ["stale", "very_stale", "outdated", "no_data", "error"]:
            st.warning("⚠️ Forecasts need updating for accurate predictions")
    
    with col2:
        if st.button("🔄 Update Now", use_container_width=True, type="primary"):
            with st.spinner("Generating fresh forecasts..."):
                if trigger_prediction_update():
                    st.success("Update triggered! Refreshing...")
                    time.sleep(3)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Failed to trigger update")
    
    with col3:
        if st.button("🔍 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Load all types of forecasts
    ml_forecast = load_ml_forecast()
    ts_forecast = load_time_series_forecast()
    ensemble_forecast = load_ensemble_forecast()
    
    # Check available forecasts
    has_ml = not ml_forecast.empty and 'predicted_aqi' in ml_forecast.columns
    has_ts = not ts_forecast.empty and 'predicted_aqi' in ts_forecast.columns
    has_ensemble = not ensemble_forecast.empty and 'predicted_aqi' in ensemble_forecast.columns
    
    if has_ml or has_ts or has_ensemble:
        
        # Show all available forecasts
        cols = st.columns(3)
        
        with cols[0]:
            st.markdown("### 🤖 ML Model Forecast")
            if has_ml:
                for _, row in ml_forecast.head(3).iterrows():
                    category, color, emoji = get_aqi_category(row['predicted_aqi'])
                    date_display = row['date'] if 'date' in row else 'Unknown'
                    if hasattr(date_display, 'strftime'):
                        date_display = date_display.strftime('%Y-%m-%d')
                    st.markdown(f"""
                    <div class="metric-card ml-card">
                        <h4>{date_display}</h4>
                        <h2 style="color: {color};">{row['predicted_aqi']:.0f}</h2>
                        <p>{emoji} {category}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No ML forecast")
        
        with cols[1]:
            st.markdown("### 📈 Time Series Forecast")
            if has_ts:
                for _, row in ts_forecast.head(3).iterrows():
                    category, color, emoji = get_aqi_category(row['predicted_aqi'])
                    date_display = row['date'] if 'date' in row else 'Unknown'
                    if hasattr(date_display, 'strftime'):
                        date_display = date_display.strftime('%Y-%m-%d')
                    st.markdown(f"""
                    <div class="metric-card ts-card">
                        <h4>{date_display}</h4>
                        <h2 style="color: {color};">{row['predicted_aqi']:.0f}</h2>
                        <p>{emoji} {category}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No time series forecast")
        
        with cols[2]:
            st.markdown("### 🎯 Ensemble Forecast")
            if has_ensemble:
                for _, row in ensemble_forecast.head(3).iterrows():
                    category, color, emoji = get_aqi_category(row['predicted_aqi'])
                    date_display = row['date'] if 'date' in row else 'Unknown'
                    if hasattr(date_display, 'strftime'):
                        date_display = date_display.strftime('%Y-%m-%d')
                    st.markdown(f"""
                    <div class="metric-card ensemble-card">
                        <h4>{date_display}</h4>
                        <h2 style="color: {color};">{row['predicted_aqi']:.0f}</h2>
                        <p>{emoji} {category}</p>
                        {'<small>Avg of ML & TS</small>' if 'ml_aqi' in row else ''}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No ensemble forecast")
        
        # Combined Forecast Chart
        st.markdown('<h2 class="sub-header">📊 Forecast Comparison</h2>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        if has_ml:
            # Prepare ML forecast data
            ml_display = ml_forecast.copy()
            if 'timestamp' in ml_display.columns:
                ml_display['date_display'] = ml_display['timestamp'].dt.date.astype(str)
            elif 'date' in ml_display.columns:
                ml_display['date_display'] = ml_display['date'].astype(str)
            
            # Get average per day
            ml_grouped = ml_display.groupby('date_display')['predicted_aqi'].mean().reset_index()
            
            fig.add_trace(go.Scatter(
                x=ml_grouped['date_display'],
                y=ml_grouped['predicted_aqi'],
                mode='lines+markers',
                name='ML Model',
                line=dict(color='#3B82F6', width=3)
            ))
        
        if has_ts and ts_forecast['source'].iloc[0] != 'Ensemble':
            # Prepare TS forecast data
            ts_display = ts_forecast.copy()
            if 'timestamp' in ts_display.columns:
                ts_display['date_display'] = ts_display['timestamp'].dt.date.astype(str)
            elif 'date' in ts_display.columns:
                ts_display['date_display'] = ts_display['date'].astype(str)
            
            ts_grouped = ts_display.groupby('date_display')['predicted_aqi'].mean().reset_index()
            
            fig.add_trace(go.Scatter(
                x=ts_grouped['date_display'],
                y=ts_grouped['predicted_aqi'],
                mode='lines+markers',
                name='Time Series',
                line=dict(color='#10B981', width=3, dash='dash')
            ))
        
        if has_ensemble:
            # Prepare ensemble forecast data
            ensemble_display = ensemble_forecast.copy()
            if 'timestamp' in ensemble_display.columns:
                ensemble_display['date_display'] = ensemble_display['timestamp'].dt.date.astype(str)
            elif 'date' in ensemble_display.columns:
                ensemble_display['date_display'] = ensemble_display['date'].astype(str)
            
            fig.add_trace(go.Scatter(
                x=ensemble_display['date_display'],
                y=ensemble_display['predicted_aqi'],
                mode='lines+markers',
                name='Ensemble (Avg)',
                line=dict(color='#8B5CF6', width=4)
            ))
        
        # Add AQI category lines
        fig.add_hline(y=50, line_dash="dot", line_color="green", 
                      annotation_text="Good", annotation_position="bottom right")
        fig.add_hline(y=100, line_dash="dot", line_color="yellow", 
                      annotation_text="Moderate", annotation_position="bottom right")
        fig.add_hline(y=150, line_dash="dot", line_color="orange", 
                      annotation_text="USG", annotation_position="bottom right")
        
        fig.update_layout(
            title='3-Day AQI Forecast Comparison',
            xaxis_title='Date',
            yaxis_title='AQI',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show raw data
        with st.expander("📋 View Forecast Data"):
            if has_ml:
                st.write("**ML Forecast:**")
                display_cols = []
                if 'date' in ml_forecast.columns:
                    display_cols.append('date')
                if 'predicted_aqi' in ml_forecast.columns:
                    display_cols.append('predicted_aqi')
                if 'category' in ml_forecast.columns:
                    display_cols.append('category')
                
                if display_cols:
                    st.dataframe(ml_forecast[display_cols].head(), use_container_width=True)
            
            if has_ts:
                st.write("**Time Series Forecast:**")
                display_cols = []
                if 'date' in ts_forecast.columns:
                    display_cols.append('date')
                if 'predicted_aqi' in ts_forecast.columns:
                    display_cols.append('predicted_aqi')
                if 'category' in ts_forecast.columns:
                    display_cols.append('category')
                
                if display_cols:
                    st.dataframe(ts_forecast[display_cols].head(), use_container_width=True)
            
            if has_ensemble:
                st.write("**Ensemble Forecast:**")
                display_cols = ['date', 'predicted_aqi', 'category']
                if 'ml_aqi' in ensemble_forecast.columns:
                    display_cols.append('ml_aqi')
                if 'ts_aqi' in ensemble_forecast.columns:
                    display_cols.append('ts_aqi')
                
                st.dataframe(ensemble_forecast[display_cols].head(), use_container_width=True)
        
        # Health recommendations based on ensemble forecast
        if has_ensemble:
            st.markdown('<h2 class="sub-header">🩺 Health Recommendations</h2>', unsafe_allow_html=True)
            
            # Get recommendations for each day
            for _, row in ensemble_forecast.head(3).iterrows():
                category = get_aqi_category(row['predicted_aqi'])[0]
                precautions = get_precautions(category)
                
                date_display = row['date'] if 'date' in row else 'Unknown'
                if hasattr(date_display, 'strftime'):
                    date_display = date_display.strftime('%Y-%m-%d')
                
                with st.expander(f"📅 {date_display} - {category} (AQI: {row['predicted_aqi']:.0f})"):
                    for precaution in precautions:
                        st.markdown(f"• {precaution}")
        
    else:
        st.info("""
        ## 📋 Forecast Training Required
        
        To generate forecasts, you need to run the combined training script:
        
        **Step 1: Run combined training**
        ```bash
        python model_training/combined_training.py
        ```
        
        Or use the orchestrator:
        ```bash
        python runallmodels.py
        ```
        
        **Step 2: Check if forecasts were generated**
        
        **Step 3: Click 'Update Now' button above to generate fresh forecasts**
        """)

# ==================== MODEL PERFORMANCE PAGE ====================
elif page == "🤖 Model Performance":
    st.markdown('<h1 class="main-header">🤖 Model Performance</h1>', unsafe_allow_html=True)
    
    metrics_data = load_model_metrics()
    
    if not metrics_data.empty:
        # Current Best Model
        st.markdown('<h2 class="sub-header">🏆 Current Best Model</h2>', unsafe_allow_html=True)
        
        best_model = metrics_data.iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model", best_model['model_name'])
        
        with col2:
            st.metric("R² Score", f"{best_model['r2_score']:.3f}")
        
        with col3:
            st.metric("MAE", f"{best_model['mae']:.2f}")
        
        with col4:
            st.metric("RMSE", f"{best_model['rmse']:.2f}")
        
        # Model Comparison
        st.markdown('<h2 class="sub-header">📊 Model Comparison</h2>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=metrics_data['model_name'],
            y=metrics_data['r2_score'],
            name='R² Score',
            marker_color='#3B82F6'
        ))
        
        fig.update_layout(
            title='Model R² Scores',
            xaxis_title='Model',
            yaxis_title='R² Score',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics Table
        st.markdown('<h2 class="sub-header">📋 Detailed Metrics</h2>', unsafe_allow_html=True)
        
        display_df = metrics_data.copy()
        display_df['last_trained'] = display_df['created_at'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(
            display_df[['model_name', 'model_type', 'r2_score', 'mae', 'rmse', 'last_trained', 'is_production']]
            .rename(columns={
                'model_name': 'Model',
                'model_type': 'Type',
                'r2_score': 'R²',
                'mae': 'MAE',
                'rmse': 'RMSE',
                'last_trained': 'Last Trained',
                'is_production': 'Production'
            }),
            use_container_width=True
        )
        
    else:
        st.info("""
        ## 🤖 No Model Metrics Available
        
        **To see model performance:**
        
        1. **Train models first:**
        ```bash
        python model_training/train_models.py
        ```
        
        or
        
        ```bash
        python model_training/combined_training.py
        ```
        
        2. **Check if models were saved to MongoDB**
        
        3. **Refresh this page** after training
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
        4. **Feature Importance** - Understand what drives AQI predictions
        5. **Health Guidance** - Provide precautions based on AQI
        6. **Data Insights** - Understand pollution patterns
        
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
        ├── model_training/       # ML models + prediction service
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
    
    # System Status
    st.markdown("### 🖥️ System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            from pymongo import MongoClient
            uri = os.getenv("MONGODB_URI")
            if uri:
                client = MongoClient(uri, serverSelectionTimeoutMS=3000)
                client.server_info()
                st.success("✅ MongoDB Connected")
                client.close()
            else:
                st.error("❌ MongoDB URI not set")
        except Exception as e:
            st.error(f"❌ MongoDB: {str(e)[:50]}")
    
    with col2:
        current_data = load_current_aqi()
        if current_data:
            st.success("✅ Data Available")
        else:
            st.warning("⚠️ No Data")
    
    with col3:
        hist_data = load_historical_data(7)
        if not hist_data.empty:
            st.success(f"✅ {len(hist_data)} records")
        else:
            st.warning("⚠️ No History")
    
    with col4:
        freshness_status, freshness_msg, _ = check_prediction_freshness()
        if freshness_status == "fresh":
            st.success("✅ Fresh")
        elif freshness_status == "stale":
            st.warning("⚠️ Stale")
        elif freshness_status == "very_stale":
            st.error("❌ Very Stale")
        else:
            st.info("ℹ️ Unknown")

# ==================== SYSTEM STATUS PAGE ====================
elif page == "⚙️ System Status":
    st.markdown('<h1 class="main-header">⚙️ System Status</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🔄 System Status
    
    Current system configuration and status.
    """)
    
    # Configuration
    st.markdown("### ⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code("""
        # .env Configuration
        MONGODB_URI=*****
        MONGODB_DATABASE=aqi_predictor
        
        # Data Collection
        CITY=Karachi
        LAT=24.8607
        LON=67.0011
        DAYS=45
        """, language="bash")
    
    with col2:
        st.info("""
        **Current Settings:**
        - Data Collection: Every 3 hours (incremental)
        - Model Training: Twice daily (3 AM & 3 PM)
        - Prediction Updates: Every 3 hours
        - Storage: MongoDB Atlas
        - Location: Karachi, Pakistan
        """)
    
    # Data Status
    st.markdown("### 📊 Data Status")
    
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if uri:
            client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            db = client[os.getenv("MONGODB_DATABASE", "aqi_predictor")]
            
            collections = db.list_collection_names()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                count = db.aqi_measurements.count_documents({}) if 'aqi_measurements' in collections else 0
                st.metric("Raw Data", f"{count} records")
            
            with col2:
                count = db.aqi_features.count_documents({}) if 'aqi_features' in collections else 0
                st.metric("Features", f"{count} records")
            
            with col3:
                model_count = db.model_registry.count_documents({}) if 'model_registry' in collections else 0
                st.metric("Models", f"{model_count} trained")
            
            # Forecast collections
            st.markdown("### 🔮 Forecast Collections")
            
            forecast_cols = st.columns(3)
            
            with forecast_cols[0]:
                count = db.ml_forecasts_3day.count_documents({}) if 'ml_forecasts_3day' in collections else 0
                st.metric("ML Forecasts", f"{count}")
            
            with forecast_cols[1]:
                count = db.timeseries_forecasts_3day.count_documents({}) if 'timeseries_forecasts_3day' in collections else 0
                st.metric("TS Forecasts", f"{count}")
            
            with forecast_cols[2]:
                count = db.ensemble_forecasts_3day.count_documents({}) if 'ensemble_forecasts_3day' in collections else 0
                st.metric("Ensemble Forecasts", f"{count}")
            
            # Prediction freshness
            st.markdown("### ⏱️ Prediction Freshness")
            
            freshness_status, freshness_msg, last_updated = check_prediction_freshness()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Status", freshness_status.upper().replace('_', ' '))
            
            with col2:
                if last_updated:
                    age_hours = (datetime.now() - last_updated).total_seconds() / 3600
                    st.metric("Age", f"{age_hours:.1f} hours")
                else:
                    st.metric("Age", "Unknown")
            
            client.close()
        else:
            st.warning("MongoDB URI not configured")
    except Exception as e:
        st.error(f"Database error: {str(e)[:50]}")
    
    # Manual Controls
    st.markdown("### 🎛️ Manual Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Collect New Data", use_container_width=True):
            st.info("To collect data, run:\n```bash\npython data_pipeline/collect_historical.py --incremental\n```")
    
    with col2:
        if st.button("🤖 Generate Predictions", use_container_width=True):
            if trigger_prediction_update():
                st.success("Prediction update triggered!")
                time.sleep(2)
                st.rerun()
            else:
                st.error("Failed to trigger update")
    
    with col3:
        if st.button("🔄 Refresh All", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

# ==================== RUN THE APP ====================
if __name__ == "__main__":
    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🌫️ AQI Karachi Prediction System | 
            <a href='https://github.com/AjiyaAnwar/aqi-Karachi' target='_blank'>GitHub</a> | 
            Dashboard: {}</p>
            <p><small>Predictions updated every 3 hours | Data collected hourly | Models trained twice daily</small></p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        unsafe_allow_html=True
    )