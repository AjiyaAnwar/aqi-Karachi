"""
📊 AQI-Karachi Dashboard: Real-time Air Quality Prediction
COMPLETE VERSION: All pages including EDA + Streamlit warnings fixed
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
warnings.filterwarnings('ignore')

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if os.path.exists(src_dir):
    sys.path.insert(0, src_dir)

load_dotenv()

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
    .aqi-good { color: #10B981; font-weight: bold; }
    .aqi-moderate { color: #F59E0B; font-weight: bold; }
    .aqi-unhealthy-sensitive { color: #F97316; font-weight: bold; }
    .aqi-unhealthy { color: #EF4444; font-weight: bold; }
    .aqi-very-unhealthy { color: #8B5CF6; font-weight: bold; }
    .aqi-hazardous { color: #7C3AED; font-weight: bold; }
    .ml-card { border-left: 5px solid #3B82F6; }
    .ts-card { border-left: 5px solid #10B981; }
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

# Data loading functions with caching
@st.cache_data(ttl=3600)
def load_current_aqi():
    """Load current AQI data from MongoDB - FIXED VERSION"""
    try:
        from pymongo import MongoClient
        
        uri = os.getenv("MONGODB_URI")
        if not uri:
            st.error("MONGODB_URI not set in .env file")
            return None
            
        db_name = os.getenv("MONGODB_DATABASE", "aqi_predictor")
        
        client = MongoClient(uri)
        db = client[db_name]
        
        # Try to find data in aqi_measurements first (your collection)
        # Then try aqi_features (dashboard expected collection)
        collections = db.list_collection_names()
        
        if 'aqi_measurements' in collections:
            # Your data collection script saves here
            latest = db.aqi_measurements.find_one(sort=[('timestamp', -1)])
            collection_name = 'aqi_measurements'
        elif 'aqi_features' in collections:
            # Dashboard expects this
            latest = db.aqi_features.find_one(sort=[('timestamp', -1)])
            collection_name = 'aqi_features'
        else:
            st.warning(f"No AQI collections found. Available collections: {collections}")
            client.close()
            return None
        
        if latest:
            # Handle different field names
            aqi_field = 'aqi' if 'aqi' in latest else 'us_aqi'
            pm25_field = 'pm25' if 'pm25' in latest else 'pm2_5'
            
            current_aqi = latest.get(aqi_field, 0)
            pm25 = latest.get(pm25_field, 0)
            pm10 = latest.get('pm10', 0)
            timestamp = latest.get('timestamp', datetime.now())
            
            # Ensure timestamp is a datetime object
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
    """Load historical data for EDA - FIXED VERSION"""
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
        
        # Try aqi_measurements first, then aqi_features
        collections = db.list_collection_names()
        
        if 'aqi_measurements' in collections:
            # Your actual data collection
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
            
            # Clean up the dataframe
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            # Ensure timestamp column exists and convert
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            # Handle different field names
            if 'us_aqi' in df.columns and 'aqi' not in df.columns:
                df['aqi'] = df['us_aqi']
            if 'pm2_5' in df.columns and 'pm25' not in df.columns:
                df['pm25'] = df['pm2_5']
            
            # Add derived features for EDA
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
        
        # Try different collection names
        forecast_collections = ['aqi_predictions', 'ml_forecasts', 'forecasts']
        
        for coll_name in forecast_collections:
            if coll_name in db.list_collection_names():
                forecast_data = list(db[coll_name].find({}, sort=[('date', 1)]))
                if forecast_data:
                    df = pd.DataFrame(forecast_data)
                    if 'date' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['date'])
                        df = df.sort_values('timestamp')
                        df['source'] = 'ML Model'
                    client.close()
                    return df
        
        client.close()
        return pd.DataFrame()
        
    except Exception as e:
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
        
        # Try different collection names
        forecast_collections = ['simple_forecasts', 'time_series_forecasts', 'ts_forecasts']
        
        for coll_name in forecast_collections:
            if coll_name in db.list_collection_names():
                forecast_data = list(db[coll_name].find({}, sort=[('timestamp', 1)]))
                if forecast_data:
                    df = pd.DataFrame(forecast_data)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values('timestamp')
                        df['source'] = 'Time Series'
                        
                        # Group by date for daily averages
                        df['date'] = df['timestamp'].dt.date
                        daily_avg = df.groupby('date').agg({
                            'predicted_aqi': 'mean'
                        }).reset_index()
                        
                        # Add timestamp for plotting
                        daily_avg['timestamp'] = pd.to_datetime(daily_avg['date'])
                        daily_avg['source'] = 'Time Series (Daily Avg)'
                        
                        client.close()
                        return daily_avg
        
        client.close()
        return pd.DataFrame()
        
    except Exception as e:
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
        
        # Try loading from model_registry
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

# Sidebar Navigation
st.sidebar.title("🌫️ AQI Karachi Dashboard")
page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "📊 Current AQI", "📈 EDA Analysis", "📊 Historical Trends", "🔮 3-Day Forecast", "🤖 Model Performance", "ℹ️ Project Info", "⚙️ CI/CD Status"]
)

st.sidebar.markdown("---")
st.sidebar.info("**Karachi AQI Prediction System**\n\nReal-time air quality forecasting using 45-day Open-Meteo data.")

# Add refresh button
if st.sidebar.button("🔄 Refresh All Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🌫️ AQI Karachi - Air Quality Prediction System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Karachi Air Quality Monitoring")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This system provides real-time air quality predictions for Karachi using:
    
    - **45 days of historical data** from Open-Meteo API
    - **Time series forecasting** for AQI predictions
    - **Machine learning models** trained on environmental factors
    
    ### 📋 Key Features:
    1. **Real-time AQI Monitoring** - Current air quality status
    2. **EDA Analysis** - Exploratory Data Analysis with visualizations
    3. **Historical Trends** - Data visualization and analysis
    4. **3-Day Forecast** - Predictive analytics for future AQI
    5. **Health Recommendations** - Precautions based on AQI levels
    6. **Model Comparison** - Performance metrics of different algorithms
    
    ### 🚀 How It Works:
    1. **Data Collection**: 45-day Open-Meteo data → MongoDB
    2. **Feature Engineering**: Extract time-based features
    3. **Model Training**: Time series and ML models
    4. **Prediction**: Generate AQI forecasts
    5. **Dashboard**: Visualize results and provide health guidance
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

# ==================== CURRENT AQI PAGE ====================
elif page == "📊 Current AQI":
    st.markdown('<h1 class="main-header">📊 Current Air Quality in Karachi</h1>', unsafe_allow_html=True)
    
    current_data = load_current_aqi()
    
    if current_data:
        # AQI Display
        aqi_value = current_data['aqi']
        category, color, emoji = get_aqi_category(aqi_value)
        
        # Format timestamp
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
        st.plotly_chart(fig, width='stretch')
        
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
        st.markdown("""
        ### 🚨 To Get Data:
        
        1. **Run data collection script:**
        ```bash
        python data_pipeline/collect_historical.py
        ```
        
        2. **Check if data was collected:**
        ```bash
        python -c "
        from pymongo import MongoClient
        import os
        from dotenv import load_dotenv
        load_dotenv()
        client = MongoClient(os.getenv('MONGODB_URI'))
        db = client[os.getenv('MONGODB_DATABASE')]
        print(f'Collections: {db.list_collection_names()}')
        print(f'aqi_measurements count: {db.aqi_measurements.count_documents({})}')
        client.close()
        "
        ```
        
        3. **Refresh this page** after data collection
        """)
# ==================== EDA ANALYSIS PAGE (UPDATED) ====================
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
        sample_size = st.slider("Sample size (for large datasets):", 1000, 10000, 5000, key="sample_size")
    
    hist_data = load_historical_data(days_to_load)
    
    if not hist_data.empty:
        # Sample data if too large for performance
        if len(hist_data) > sample_size:
            hist_data = hist_data.sample(sample_size, random_state=42)
            st.info(f"📊 Using sample of {sample_size} records for better performance")
        
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
            st.dataframe(col_info, width='stretch')
            
            # Data Sample
            with st.expander("👀 View Data Sample"):
                st.dataframe(hist_data.head(20), width='stretch')
            
            # Statistical Summary
            st.markdown("### 📈 Statistical Summary")
            numeric_cols = hist_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary_stats = hist_data[numeric_cols].describe().T
                summary_stats['skew'] = hist_data[numeric_cols].skew()
                summary_stats['kurtosis'] = hist_data[numeric_cols].kurtosis()
                st.dataframe(summary_stats, width='stretch')
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
                    st.plotly_chart(fig1, width='stretch')
                
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
                    st.plotly_chart(fig2, width='stretch')
                
                # AQI Categories Distribution
                st.markdown("### 🏷️ AQI Categories Distribution")
                
                # Categorize AQI
                def categorize_aqi_value(x):
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
                    "Hazardous": "#7C3AED"
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig3 = px.pie(category_counts, values='Count', names='Category',
                                 title='AQI Category Distribution',
                                 color='Category',
                                 color_discrete_map=category_colors)
                    fig3.update_layout(height=400)
                    st.plotly_chart(fig3, width='stretch')
                
                with col2:
                    # Bar chart
                    fig4 = px.bar(category_counts, x='Category', y='Percentage',
                                 title='AQI Category Percentage',
                                 color='Category',
                                 color_discrete_map=category_colors,
                                 text='Percentage')
                    fig4.update_layout(height=400, yaxis_title='Percentage (%)')
                    st.plotly_chart(fig4, width='stretch')
                
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
                st.plotly_chart(fig5, width='stretch')
                
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
                st.plotly_chart(fig1, width='stretch')
                
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
                st.plotly_chart(fig2, width='stretch')
                
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
                st.plotly_chart(fig3, width='stretch')
                
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
                st.plotly_chart(fig4, width='stretch')
                
                # Rolling Averages
                st.markdown("### 📈 Rolling Averages")
                hist_data_sorted = hist_data.sort_values('timestamp')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 7-day rolling average
                    rolling_7d = hist_data_sorted.set_index('timestamp')['aqi'].rolling(window=7).mean()
                    fig5 = go.Figure()
                    fig5.add_trace(go.Scatter(x=rolling_7d.index, y=rolling_7d.values,
                                             mode='lines',
                                             name='7-Day Rolling Avg',
                                             line=dict(color='red', width=2)))
                    fig5.update_layout(title='7-Day Rolling Average',
                                      xaxis_title='Date',
                                      yaxis_title='AQI',
                                      height=300)
                    st.plotly_chart(fig5, width='stretch')
                
                with col2:
                    # 24-hour rolling average
                    rolling_24h = hist_data_sorted.set_index('timestamp')['aqi'].rolling(window=24).mean()
                    fig6 = go.Figure()
                    fig6.add_trace(go.Scatter(x=rolling_24h.index, y=rolling_24h.values,
                                             mode='lines',
                                             name='24-Hour Rolling Avg',
                                             line=dict(color='green', width=2)))
                    fig6.update_layout(title='24-Hour Rolling Average',
                                      xaxis_title='Date',
                                      yaxis_title='AQI',
                                      height=300)
                    st.plotly_chart(fig6, width='stretch')
                
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
                st.plotly_chart(fig1, width='stretch')
                
                # Top correlations with AQI
                if 'aqi' in numeric_cols:
                    st.markdown("### 🔝 Top Correlations with AQI")
                    aqi_corr = corr_matrix['aqi'].drop('aqi', errors='ignore').sort_values(ascending=False)
                    
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=aqi_corr.head(15).values,
                        y=aqi_corr.head(15).index,
                        orientation='h',
                        marker_color=['green' if x > 0 else 'red' for x in aqi_corr.head(15).values]
                    ))
                    fig2.update_layout(title='Top 15 Features Correlated with AQI',
                                      xaxis_title='Correlation Coefficient',
                                      yaxis_title='Feature',
                                      height=400)
                    st.plotly_chart(fig2, width='stretch')
                    
                    # Scatter plots for top correlated features
                    st.markdown("### 📊 Scatter Plots with AQI")
                    top_features = aqi_corr.head(4).index.tolist()
                    
                    if len(top_features) > 0:
                        cols = st.columns(2)
                        for i, feature in enumerate(top_features[:4]):
                            with cols[i % 2]:
                                fig = px.scatter(hist_data, x=feature, y='aqi',
                                               trendline='ols',
                                               title=f'{feature} vs AQI',
                                               opacity=0.5)
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, width='stretch')
                    
                    # Correlation statistics
                    with st.expander("📈 View Correlation Statistics"):
                        st.write("**Correlation with AQI:**")
                        corr_stats = pd.DataFrame({
                            'Feature': aqi_corr.index,
                            'Correlation': aqi_corr.values,
                            'Strength': ['Strong' if abs(x) > 0.7 else 'Moderate' if abs(x) > 0.3 else 'Weak' 
                                        for x in aqi_corr.values]
                        })
                        st.dataframe(corr_stats, width='stretch')
                
                # Pairplot for top features
                st.markdown("### 🎨 Pairwise Relationships")
                if 'aqi' in numeric_cols and len(numeric_cols) >= 4:
                    top_4 = ['aqi'] + aqi_corr.head(3).index.tolist() if 'aqi_corr' in locals() else numeric_cols[:4]
                    
                    # Create a scatter matrix
                    import plotly.figure_factory as ff
                    
                    fig = ff.create_scatterplotmatrix(
                        hist_data[top_4],
                        diag='histogram',
                        height=600,
                        title='Scatter Matrix of Top Features'
                    )
                    st.plotly_chart(fig, width='stretch')
            else:
                st.warning("Not enough numeric columns for correlation analysis")
        
        # ========== TAB 5: OUTLIERS & ANOMALIES ==========
        with tab5:
            if 'aqi' in hist_data.columns:
                st.markdown('<h2 class="sub-header">🔍 Outlier & Anomaly Detection</h2>', unsafe_allow_html=True)
                
                aqi_data = hist_data['aqi'].dropna()
                
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
                st.plotly_chart(fig1, width='stretch')
                
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
                    st.plotly_chart(fig2, width='stretch')
                
                # View outliers table
                with st.expander("📋 View Detected Outliers"):
                    if len(outliers) > 0:
                        outliers_display = outliers.copy()
                        if 'timestamp' in outliers_display.columns:
                            outliers_display['timestamp'] = outliers_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                        st.dataframe(outliers_display[['timestamp', 'aqi']].head(20), width='stretch')
                    else:
                        st.info("No outliers detected")
                
                # Distribution of outlier categories
                if len(outliers) > 0:
                    st.markdown("### 🏷️ Outlier Categories")
                    outliers['aqi_category'] = outliers['aqi'].apply(categorize_aqi_value)
                    outlier_categories = outliers['aqi_category'].value_counts().reset_index()
                    outlier_categories.columns = ['Category', 'Count']
                    
                    fig3 = px.bar(outlier_categories, x='Category', y='Count',
                                 title='Outlier Distribution by AQI Category',
                                 color='Category',
                                 color_discrete_map=category_colors)
                    fig3.update_layout(height=300)
                    st.plotly_chart(fig3, width='stretch')
                
                # Z-score analysis
                st.markdown("### 📊 Z-Score Analysis")
                from scipy import stats
                z_scores = np.abs(stats.zscore(aqi_data))
                extreme_outliers = aqi_data[z_scores > 3]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Z-score > 3", len(extreme_outliers))
                with col2:
                    extreme_percent = (len(extreme_outliers) / len(aqi_data)) * 100
                    st.metric("Percentage", f"{extreme_percent:.2f}%")
            
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
                    st.metric("Skewness", f"{aqi_data.skew():.2f}")
                    st.metric("Kurtosis", f"{aqi_data.kurtosis():.2f}")
                
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
                skew_val = aqi_data.skew()
                if abs(skew_val) > 1:
                    skew_direction = "right" if skew_val > 0 else "left"
                    insights.append(f"📊 **Distribution**: AQI is highly skewed to the {skew_direction} (skewness = {skew_val:.2f})")
                elif abs(skew_val) > 0.5:
                    skew_direction = "right" if skew_val > 0 else "left"
                    insights.append(f"📊 **Distribution**: AQI is moderately skewed to the {skew_direction}")
                else:
                    insights.append("✅ **Distribution**: AQI is approximately symmetric")
                
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
                    if st.button("📥 Download Summary", use_container_width=True):
                        # Create summary report
                        summary_text = f"""
                        AQI Karachi - EDA Summary Report
                        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        
                        Dataset Summary:
                        - Records: {len(hist_data):,}
                        - Time Period: {days_to_load} days
                        - Avg AQI: {aqi_data.mean():.1f}
                        - Std Dev: {aqi_data.std():.1f}
                        - Missing Data: {completeness:.1f}% complete
                        
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
        st.markdown("""
        ### 🚨 To Get EDA Data:
        
        1. **Run data collection:**
        ```bash
        python data_pipeline/collect_historical.py
        ```
        
        2. **Refresh this page** after data collection
        
        3. **Check MongoDB connection** in .env file
        """)

# ==================== HISTORICAL TRENDS PAGE ====================
elif page == "📊 Historical Trends":
    st.markdown('<h1 class="main-header">📊 Historical AQI Trends</h1>', unsafe_allow_html=True)
    
    # Load data with adjustable days
    days_to_load = st.slider("Select number of days to analyze:", 7, 90, 30, key="history_days")
    hist_data = load_historical_data(days_to_load)
    
    if not hist_data.empty:
        # Time Series Plot
        st.markdown('<h2 class="sub-header">⏰ AQI Time Series</h2>', unsafe_allow_html=True)
        
        if 'timestamp' in hist_data.columns and 'aqi' in hist_data.columns:
            fig = px.line(hist_data, x='timestamp', y='aqi',
                         title=f'AQI Trend - Last {days_to_load} Days',
                         labels={'aqi': 'AQI', 'timestamp': 'Time'},
                         line_shape='spline')
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Timestamp or AQI column not found in data")
        
        # Daily Averages
        st.markdown('<h2 class="sub-header">📅 Daily Averages</h2>', unsafe_allow_html=True)
        
        if 'date' in hist_data.columns and 'aqi' in hist_data.columns:
            daily_avg = hist_data.groupby('date')['aqi'].agg(['mean', 'min', 'max']).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_avg['date'],
                y=daily_avg['mean'],
                mode='lines+markers',
                name='Daily Avg',
                line=dict(color='#3B82F6', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=daily_avg['date'],
                y=daily_avg['min'],
                mode='lines',
                name='Daily Min',
                line=dict(color='#10B981', width=1, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=daily_avg['date'],
                y=daily_avg['max'],
                mode='lines',
                name='Daily Max',
                line=dict(color='#EF4444', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title='Daily AQI Statistics',
                xaxis_title='Date',
                yaxis_title='AQI',
                height=400
            )
            st.plotly_chart(fig, width='stretch')
        
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
            st.plotly_chart(fig, width='stretch')
        
    else:
        st.warning("No historical data available.")

# ==================== 3-DAY FORECAST PAGE ====================
elif page == "🔮 3-Day Forecast":
    st.markdown('<h1 class="main-header">🔮 3-Day AQI Forecast</h1>', unsafe_allow_html=True)
    
    # Load forecasts
    ml_forecast = load_ml_forecast()
    ts_forecast = load_time_series_forecast()
    
    # Check if we have forecast data
    has_ml_forecast = not ml_forecast.empty and 'predicted_aqi' in ml_forecast.columns
    has_ts_forecast = not ts_forecast.empty and 'predicted_aqi' in ts_forecast.columns
    
    if has_ml_forecast or has_ts_forecast:
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 ML Model Forecast")
            if has_ml_forecast:
                ml_forecast['date'] = ml_forecast['timestamp'].dt.date
                daily_ml = ml_forecast.groupby('date').agg({
                    'predicted_aqi': 'mean'
                }).round(1).reset_index()
                
                for _, row in daily_ml.head(3).iterrows():
                    category, color, emoji = get_aqi_category(row['predicted_aqi'])
                    st.markdown(f"""
                    <div class="metric-card ml-card">
                        <h4>{row['date']}</h4>
                        <h2 style="color: {color};">{row['predicted_aqi']:.0f}</h2>
                        <p>{emoji} {category}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No ML forecast available")
        
        with col2:
            st.markdown("### 📈 Time Series Forecast")
            if has_ts_forecast:
                for _, row in ts_forecast.head(3).iterrows():
                    category, color, emoji = get_aqi_category(row['predicted_aqi'])
                    st.markdown(f"""
                    <div class="metric-card ts-card">
                        <h4>{row['date']}</h4>
                        <h2 style="color: {color};">{row['predicted_aqi']:.0f}</h2>
                        <p>{emoji} {category}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No time series forecast available")
        
        # Combined Forecast Chart
        st.markdown('<h2 class="sub-header">📊 Forecast Comparison</h2>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        if has_ml_forecast:
            fig.add_trace(go.Scatter(
                x=ml_forecast['timestamp'],
                y=ml_forecast['predicted_aqi'],
                mode='lines+markers',
                name='ML Model',
                line=dict(color='#3B82F6', width=3)
            ))
        
        if has_ts_forecast:
            fig.add_trace(go.Scatter(
                x=ts_forecast['timestamp'],
                y=ts_forecast['predicted_aqi'],
                mode='lines+markers',
                name='Time Series',
                line=dict(color='#10B981', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title='3-Day AQI Forecast',
            xaxis_title='Date',
            yaxis_title='AQI',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, width='stretch')
        
    else:
        st.info("""
        ## 📋 Forecast Training Required
        
        To generate forecasts, you need to:
        
        1. **Collect historical data** (already done)
        2. **Train ML models** 
        3. **Run time series forecasting**
        
        ### 🚀 Quick Start:
        
        **Step 1: Check your data**
        ```bash
        python -c "
        from pymongo import MongoClient
        import os
        from dotenv import load_dotenv
        load_dotenv()
        client = MongoClient(os.getenv('MONGODB_URI'))
        db = client[os.getenv('MONGODB_DATABASE')]
        count = db.aqi_measurements.count_documents({})
        print(f'You have {count} AQI records')
        if count > 0:
            latest = db.aqi_measurements.find_one(sort=[('timestamp', -1)])
            print(f'Latest AQI: {latest.get(\"aqi\", \"N/A\")}')
        client.close()
        "
        ```
        
        **Step 2: Train models** (when ready)
        ```bash
        python model_training/train_models.py
        ```
        
        **Step 3: Refresh dashboard** after training
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
        
        st.plotly_chart(fig, width='stretch')
        
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
            width='stretch'
        )
        
    else:
        st.info("""
        ## 🤖 No Model Metrics Available
        
        **To see model performance:**
        
        1. **Train ML models first:**
        ```bash
        python model_training/train_models.py
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
        1. **Real-time Monitoring** - Track current air quality
        2. **Predictive Analytics** - Forecast AQI for next 3 days
        3. **EDA Analysis** - Exploratory data analysis with visualizations
        4. **Health Guidance** - Provide precautions based on AQI
        5. **Data Insights** - Understand pollution patterns
        
        ### 🛠️ Technology Stack:
        - **Backend**: Python, MongoDB, Open-Meteo API
        - **ML Models**: Random Forest, XGBoost, etc.
        - **Dashboard**: Streamlit, Plotly
        - **Data Pipeline**: Automated collection & processing
        
        ### 📊 Data Sources:
        1. **Open-Meteo API** - Historical weather & air quality
        2. **45-day historical data** - For model training
        
        ### 🔄 Pipeline:
        1. **Data Collection** → MongoDB
        2. **Feature Engineering** → Time-based features
        3. **Model Training** → ML + Time Series
        4. **Prediction** → 3-day forecasts
        5. **Dashboard** → Visualization & alerts
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
        2. Run data collection
        3. Launch dashboard
        4. Train models (optional)
        """)
    
    # System Status
    st.markdown("### 🖥️ System Status")
    
    col1, col2, col3 = st.columns(3)
    
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

# ==================== CI/CD STATUS PAGE ====================
elif page == "⚙️ CI/CD Status":
    st.markdown('<h1 class="main-header">⚙️ System Status</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🔄 Data Pipeline Status
    
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
        - Data Collection: 45 days
        - Update Frequency: Manual
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
            
            client.close()
        else:
            st.warning("MongoDB URI not configured")
    except Exception as e:
        st.error(f"Database error: {str(e)[:50]}")
    
    # Manual Controls
    st.markdown("### 🎛️ Manual Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Run Data Collection", use_container_width=True):
            st.info("To collect data, run:\n```bash\npython data_pipeline/collect_historical.py\n```")
    
    with col2:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        if st.button("📋 View Collections", use_container_width=True):
            try:
                from pymongo import MongoClient
                uri = os.getenv("MONGODB_URI")
                if uri:
                    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
                    db = client[os.getenv("MONGODB_DATABASE", "aqi_predictor")]
                    collections = db.list_collection_names()
                    st.write("**Collections:**", collections)
                    for coll in collections:
                        count = db[coll].count_documents({})
                        st.write(f"- {coll}: {count} records")
                    client.close()
            except Exception as e:
                st.error(f"Error: {e}")

# ==================== RUN THE APP ====================
if __name__ == "__main__":
    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🌫️ AQI Karachi Prediction System | 
            <a href='https://github.com/AjiyaAnwar/aqi-Karachi' target='_blank'>GitHub</a> | 
            Updated: {}</p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        unsafe_allow_html=True
    )